#!/usr/bin/env python3
"""
Evaluate a trained Qwen3-VL gaze policy (LoRA) on COCO-Search18.

We roll out the model for up to N fixations per trial and grade it as a
present/absent classifier:
  pred_present  := model outputs FOUND
  pred_absent   := model outputs NOT_FOUND (or fails to finalize)

This is a coarse sanity metric: "is it better than random/majority guess?"

Example:
  uv run -m saccade.eval_rollout_qwen3vl_gaze --split valid --condition all --max-examples 200
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, set_seed

from saccade.coco_search18_dataset import ORIG_H, ORIG_W, COCOSearch18Dataset
from saccade.model import FINAL_FOUND, FINAL_NOT_FOUND, encode_messages
from saccade.rollout_qwen3vl_gaze import _crop_patch_rgb, _default_system_prompt, _parse_reply, _read_base_model_id


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--model-dir",
        type=Path,
        default=Path("runs/qwen3vl-gaze-lora"),
        help="Directory with `adapter_model.safetensors` and tokenizer/processor files.",
    )
    p.add_argument(
        "--base-model-id",
        type=str,
        default=None,
        help="Override the base model id. If unset, read from adapter_config.json.",
    )
    p.add_argument("--merge-lora", action="store_true")

    p.add_argument("--data-dir", type=Path, default=Path("data/coco_search18"))
    p.add_argument("--split", default="valid", choices=["train", "valid", "all"])
    p.add_argument("--condition", default="all", choices=["TP", "TA", "all"])

    p.add_argument("--patch-size", type=int, nargs=2, default=(224, 224), metavar=("H", "W"))
    p.add_argument("--max-fixations", type=int, default=12)
    p.add_argument("--no-resize", action="store_true", help=f"Do not resize to {ORIG_W}x{ORIG_H}.")
    p.add_argument(
        "--force-final",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If no final decision during rollout, do a separate decision pass to get FOUND/NOT_FOUND.",
    )

    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)

    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--out-json", type=Path, default=None, help="Write a JSON summary here.")
    p.add_argument("--out-jsonl", type=Path, default=None, help="Write per-trial predictions here (jsonl).")

    return p.parse_args()


def _rollout_one(
    *,
    model,
    processor,
    full_rgb: Image.Image,
    target: str,
    start_xy: tuple[int, int],
    patch_size: tuple[int, int],
    max_fixations: int,
    max_new_tokens: int,
    sample: bool,
    temperature: float,
    top_p: float,
    device: torch.device,
    dtype: torch.dtype,
    force_final: bool,
) -> dict[str, Any]:
    W, H = full_rgb.size

    fixations: list[tuple[int, int]] = [(int(start_xy[0]), int(start_xy[1]))]
    patches: list[Image.Image] = []
    outputs: list[str] = []
    stop_reason = "max_fixations"
    final_text: str | None = None

    system_prompt = _default_system_prompt(target)

    for _t in range(int(max_fixations)):
        cx, cy = fixations[-1]
        cx = int(max(0, min(cx, W - 1)))
        cy = int(max(0, min(cy, H - 1)))

        patch = _crop_patch_rgb(full_rgb, cx=cx, cy=cy, patch_size=patch_size)
        patches.append(patch)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
        ]
        for p_img, out_text in zip(patches[:-1], outputs):
            messages.append({"role": "user", "content": [{"type": "image", "image": p_img}]})
            messages.append({"role": "assistant", "content": [{"type": "text", "text": out_text}]})

        user_content = [{"type": "image", "image": patches[-1]}]
        # On the last allowed turn, optionally nudge the model to terminate.
        if force_final and len(patches) == int(max_fixations):
            user_content.append({"type": "text", "text": f"\nFinal turn. Reply with {FINAL_FOUND} or {FINAL_NOT_FOUND}."})
        messages.append({"role": "user", "content": user_content})

        batch = encode_messages(processor, messages, add_generation_prompt=True, device=device)
        in_len = int(batch.inputs["input_ids"].shape[1])

        gen_kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": bool(sample),
            "pad_token_id": processor.tokenizer.pad_token_id,
            "eos_token_id": processor.tokenizer.eos_token_id,
        }
        if sample:
            gen_kwargs["temperature"] = float(temperature)
            gen_kwargs["top_p"] = float(top_p)

        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=dtype):
            out_ids = model.generate(**batch.inputs, **gen_kwargs)

        new_ids = out_ids[0, in_len:]
        text = processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        outputs.append(text)

        kind, val = _parse_reply(text)
        if kind == "final":
            stop_reason = "final"
            final_text = str(val)
            break
        if kind == "xy":
            x1, y1 = map(int, val)
            x1 = int(max(0, min(x1, W - 1)))
            y1 = int(max(0, min(y1, H - 1)))
            if len(fixations) >= int(max_fixations):
                stop_reason = "max_fixations"
                final_text = None
                break
            fixations.append((x1, y1))
            continue

        stop_reason = "parse_error"
        final_text = None
        break

    if final_text is None and force_final:
        # Do a separate "decision" pass that *only* asks for FOUND/NOT_FOUND based on
        # the patches seen. This avoids priming the model into spitting out more "x y".
        decision_prompt = (
            "You are evaluating a visual search episode.\n"
            f"Target category: {target}\n"
            "You will be shown the sequence of fovea patches visited.\n"
            f"Reply with exactly {FINAL_FOUND} if the target is present, otherwise {FINAL_NOT_FOUND}.\n"
            "Do not output coordinates."
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": decision_prompt}]}
        ]
        for i, p_img in enumerate(patches):
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": p_img},
                        {"type": "text", "text": f"\npatch_index={i}"},
                    ],
                }
            )
        messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": f"Answer with exactly {FINAL_FOUND} or {FINAL_NOT_FOUND}."}],
            }
        )

        batch = encode_messages(processor, messages, add_generation_prompt=True, device=device)
        in_len = int(batch.inputs["input_ids"].shape[1])
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=dtype):
            out_ids = model.generate(
                **batch.inputs,
                max_new_tokens=min(int(max_new_tokens), 8),
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
        new_ids = out_ids[0, in_len:]
        text = processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        outputs.append(text)
        kind, val = _parse_reply(text)
        if kind == "final":
            stop_reason = stop_reason + "+forced_decision"
            final_text = str(val)
        else:
            stop_reason = stop_reason + "+forced_decision_parse_error"

    return {
        "stop_reason": stop_reason,
        "final_text": final_text,
        "n_fixations": int(len(fixations)),
    }


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else float("nan")


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Install a CUDA build of torch.")

    device = torch.device(args.device)
    dtype = (
        torch.bfloat16
        if (device.type == "cuda" and torch.cuda.is_bf16_supported())
        else torch.float16
    )

    model_dir = Path(args.model_dir)
    base_model_id = args.base_model_id or _read_base_model_id(model_dir) or "Qwen/Qwen3-VL-2B-Instruct"

    processor = AutoProcessor.from_pretrained(str(model_dir))
    base = Qwen3VLForConditionalGeneration.from_pretrained(base_model_id, torch_dtype=dtype)
    model = PeftModel.from_pretrained(base, str(model_dir))
    if args.merge_lora and hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()
    model.to(device)
    model.eval()

    ds = COCOSearch18Dataset(
        args.data_dir,
        split=args.split,
        condition=args.condition,
        patch_size=tuple(args.patch_size),
    )

    idxs = np.arange(len(ds.trials), dtype=np.int64)
    if args.shuffle:
        rng = np.random.default_rng(int(args.seed))
        rng.shuffle(idxs)
    if args.max_examples is not None:
        idxs = idxs[: int(args.max_examples)]

    stop_counts: Counter[str] = Counter()
    # Confusion for y_true in {0:absent, 1:present}, y_pred in same.
    tp = fp = tn = fn = 0
    n_present = n_absent = 0
    n_finalize = 0
    n_finalize_correct = 0
    invalid_present = invalid_absent = 0

    out_f = None
    if args.out_jsonl is not None:
        args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        out_f = open(args.out_jsonl, "w")

    try:
        for j, i in enumerate(idxs.tolist()):
            sp = ds.trials[int(i)]

            image_path = ds._get_image_path(sp)
            target = str(sp["task"])
            x0 = int(round(float(sp["X"][0])))
            y0 = int(round(float(sp["Y"][0])))
            true_present = bool(sp.get("condition") == "present")

            if true_present:
                n_present += 1
            else:
                n_absent += 1

            full_rgb = Image.open(image_path).convert("RGB")
            if not args.no_resize:
                full_rgb = full_rgb.resize((ORIG_W, ORIG_H), resample=Image.BICUBIC)

            r = _rollout_one(
                model=model,
                processor=processor,
                full_rgb=full_rgb,
                target=target,
                start_xy=(x0, y0),
                patch_size=tuple(args.patch_size),
                max_fixations=int(args.max_fixations),
                max_new_tokens=int(args.max_new_tokens),
                sample=bool(args.sample),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                device=device,
                dtype=dtype,
                force_final=bool(args.force_final),
            )

            stop_reason = str(r["stop_reason"])
            final_text = r["final_text"]
            stop_counts[stop_reason] += 1

            finalized = bool(final_text in {FINAL_FOUND, FINAL_NOT_FOUND})
            if finalized:
                n_finalize += 1
                pred_present = bool(final_text == FINAL_FOUND)
                if bool(pred_present) == bool(true_present):
                    n_finalize_correct += 1
            else:
                pred_present = False  # treat invalid/no-final as NOT_FOUND
                if true_present:
                    invalid_present += 1
                else:
                    invalid_absent += 1

            if true_present and pred_present:
                tp += 1
            elif (not true_present) and pred_present:
                fp += 1
            elif (not true_present) and (not pred_present):
                tn += 1
            else:
                fn += 1

            if out_f is not None:
                row = {
                    "idx": int(i),
                    "name": str(sp.get("name")),
                    "target": target,
                    "true_condition": "present" if true_present else "absent",
                    "pred_present": bool(pred_present),
                    "final_text": final_text,
                    "stop_reason": stop_reason,
                    "n_fixations": int(r["n_fixations"]),
                }
                out_f.write(json.dumps(row) + "\n")

            if (j + 1) % 25 == 0:
                print(f"[{j+1}/{len(idxs)}] acc={(tp+tn)/(j+1):.3f} finalize={n_finalize/(j+1):.3f}")
    finally:
        if out_f is not None:
            out_f.close()

    n_total = int(len(idxs))
    p_present = _safe_div(n_present, n_total)
    p_absent = _safe_div(n_absent, n_total)

    acc = _safe_div(tp + tn, n_total)
    acc_present = _safe_div(tp, n_present)
    acc_absent = _safe_div(tn, n_absent)
    bal_acc = (acc_present + acc_absent) / 2.0

    prec_present = _safe_div(tp, tp + fp)
    rec_present = _safe_div(tp, tp + fn)
    f1_present = _safe_div(2.0 * prec_present * rec_present, prec_present + rec_present)

    majority_acc = max(p_present, p_absent)
    uniform_random_acc = 0.5 if (n_present and n_absent) else 1.0
    prior_random_acc = (p_present * p_present) + (p_absent * p_absent)

    summary = {
        "model_dir": str(model_dir),
        "base_model_id": str(base_model_id),
        "split": str(args.split),
        "condition": str(args.condition),
        "n_total": n_total,
        "n_present": int(n_present),
        "n_absent": int(n_absent),
        "present_rate": float(p_present),
        "finalize_rate": _safe_div(n_finalize, n_total),
        "finalize_accuracy": _safe_div(n_finalize_correct, n_finalize),
        "invalid_present": int(invalid_present),
        "invalid_absent": int(invalid_absent),
        "stop_reasons": dict(stop_counts),
        "confusion": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
        "metrics": {
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "present_recall": float(acc_present),  # TPR
            "absent_recall": float(acc_absent),  # TNR
            "present_precision": float(prec_present),
            "present_f1": float(f1_present),
        },
        "baselines": {
            "majority_accuracy": float(majority_acc),
            "uniform_random_accuracy": float(uniform_random_acc),
            "prior_random_accuracy": float(prior_random_acc),
        },
        "rollout": {
            "max_fixations": int(args.max_fixations),
            "patch_size": [int(args.patch_size[0]), int(args.patch_size[1])],
            "force_final": bool(args.force_final),
            "sample": bool(args.sample),
            "max_new_tokens": int(args.max_new_tokens),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
        },
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "seed": int(args.seed),
    }

    print(json.dumps(summary, indent=2))

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
