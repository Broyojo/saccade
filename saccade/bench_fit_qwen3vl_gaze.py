#!/usr/bin/env python3
"""
VRAM fit benchmark for Qwen3-VL gaze SFT (LoRA).

This runs:
  - 1 train step (forward + backward + AdamW step)
  - 1 eval step (forward only)
on the *longest* train/valid trials with length <= --max-fixations.

Use this to find a safe (max_fixations, lora_r) combination for your GPU.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, set_seed

from saccade.coco_search18_dataset import COCOSearch18Dataset
from saccade.train_sft_qwen3vl_gaze import Qwen3VLGazeCollator


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/coco_search18"))
    p.add_argument("--model-id", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--condition", choices=["TP", "TA", "all"], default="all")
    p.add_argument("--patch-size", type=int, nargs=2, default=(224, 224), metavar=("H", "W"))

    p.add_argument("--max-fixations", type=int, default=12)
    p.add_argument("--boxed-final", action="store_true")
    p.add_argument("--batch-size", type=int, default=1, help="Micro-batch size to test (duplicates the worst-case item).")

    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--lora-target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def _build_best_index_by_length(ds: COCOSearch18Dataset) -> dict[int, int]:
    best: dict[int, int] = {}
    for i, sp in enumerate(ds.trials):
        l = int(sp.get("length", 0))
        if l < 2:
            continue
        # First index for each length is fine; we only need a representative.
        best.setdefault(l, i)
    return best


def _pick_longest(best_by_len: dict[int, int], max_fixations: int) -> tuple[int, int]:
    for l in range(int(max_fixations), 1, -1):
        idx = best_by_len.get(l)
        if idx is not None:
            return idx, l
    raise RuntimeError(f"No trials with length <= {max_fixations} found.")


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _gb(x_bytes: int) -> float:
    return float(x_bytes) / (1024.0**3)


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Install a CUDA build of torch.")

    device = torch.device(args.device)
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    processor = AutoProcessor.from_pretrained(args.model_id)
    collator = Qwen3VLGazeCollator(processor=processor, boxed_final=args.boxed_final)

    train_base = COCOSearch18Dataset(
        args.data_dir,
        split="train",
        condition=args.condition,
        patch_size=tuple(args.patch_size),
    )
    valid_base = COCOSearch18Dataset(
        args.data_dir,
        split="valid",
        condition=args.condition,
        patch_size=tuple(args.patch_size),
    )

    train_best = _build_best_index_by_length(train_base)
    valid_best = _build_best_index_by_length(valid_base)

    train_idx, train_len = _pick_longest(train_best, args.max_fixations)
    valid_idx, valid_len = _pick_longest(valid_best, args.max_fixations)

    train_item = train_base[train_idx]
    valid_item = valid_base[valid_idx]

    train_batch = collator([train_item] * int(args.batch_size))
    valid_batch = collator([valid_item] * int(args.batch_size))

    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_id, torch_dtype=dtype)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora_targets = [s.strip() for s in args.lora_target_modules.split(",") if s.strip()]
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=lora_targets,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Train step.
    torch.cuda.empty_cache() if device.type == "cuda" else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    train_batch = _to_device(train_batch, device)
    with torch.autocast(device_type=device.type, dtype=dtype):
        out = model(**train_batch)
        loss = out.loss
    loss.backward()
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
        train_peak_alloc = int(torch.cuda.max_memory_allocated())
        train_peak_reserved = int(torch.cuda.max_memory_reserved())
    else:
        train_peak_alloc = train_peak_reserved = 0

    # Eval step.
    torch.cuda.empty_cache() if device.type == "cuda" else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.eval()
    valid_batch = _to_device(valid_batch, device)
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype):
        out2 = model(**valid_batch)
        eval_loss = out2.loss
    if device.type == "cuda":
        torch.cuda.synchronize()
        eval_peak_alloc = int(torch.cuda.max_memory_allocated())
        eval_peak_reserved = int(torch.cuda.max_memory_reserved())
        total_mem = int(torch.cuda.get_device_properties(device).total_memory)
    else:
        eval_peak_alloc = eval_peak_reserved = total_mem = 0

    result = {
        "ok": True,
        "model_id": args.model_id,
        "condition": args.condition,
        "patch_size": list(args.patch_size),
        "max_fixations": int(args.max_fixations),
        "train_len": int(train_len),
        "valid_len": int(valid_len),
        "batch_size": int(args.batch_size),
        "lora_r": int(args.lora_r),
        "lora_alpha": int(args.lora_alpha),
        "train_loss": float(loss.detach().cpu().item()),
        "eval_loss": float(eval_loss.detach().cpu().item()),
        "train_peak_alloc_gb": _gb(train_peak_alloc),
        "train_peak_reserved_gb": _gb(train_peak_reserved),
        "eval_peak_alloc_gb": _gb(eval_peak_alloc),
        "eval_peak_reserved_gb": _gb(eval_peak_reserved),
        "total_mem_gb": _gb(total_mem),
        "headroom_gb": _gb(max(total_mem - max(train_peak_reserved, eval_peak_reserved), 0)),
        "dtype": str(dtype).replace("torch.", ""),
    }
    print(json.dumps(result))

    # Help CUDA allocator reclaim memory before the process exits.
    del model, optimizer, train_batch, valid_batch, train_item, valid_item
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    try:
        main()
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        # "RuntimeError: CUDA out of memory" isn't always wrapped.
        msg = str(e)
        if "out of memory" not in msg.lower():
            raise
        print(json.dumps({"ok": False, "error": msg}))
