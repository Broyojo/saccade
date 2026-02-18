#!/usr/bin/env python3
"""
Fast, batched present/absent evaluation using vLLM (+ optional LoRA).

This is NOT a rollout evaluator. It feeds the *ground-truth patch sequence*
from COCO-Search18 and asks the model for a final decision:
  FOUND     => predict present (TP)
  NOT_FOUND => predict absent  (TA)

Example:
  uv run -m saccade.eval_vllm_present_absent_qwen3vl_gaze --split valid --condition all --max-examples 512
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

from saccade.coco_search18_dataset import COCOSearch18Dataset
from saccade.model import FINAL_FOUND, FINAL_NOT_FOUND, patch_tensor_to_pil
from saccade.rollout_qwen3vl_gaze import _parse_reply


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--data-dir", type=Path, default=Path("data/coco_search18"))
    p.add_argument("--split", default="valid", choices=["train", "valid", "all"])
    p.add_argument("--condition", default="all", choices=["TP", "TA", "all"])
    p.add_argument("--patch-size", type=int, nargs=2, default=(224, 224), metavar=("H", "W"))
    p.add_argument("--max-fixations", type=int, default=12, help="Max patches to feed per trial.")

    p.add_argument("--base-model-id", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument(
        "--lora-dir",
        type=Path,
        default=None,
        help="Optional PEFT LoRA adapter directory (e.g. runs/qwen3vl-gaze-lora).",
    )
    p.add_argument("--max-lora-rank", type=int, default=256)

    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--gpu-mem-util", type=float, default=0.85)

    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--out-jsonl", type=Path, default=None)

    return p.parse_args()


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else float("nan")


def _build_decision_prompt(*, target: str, n_patches: int) -> str:
    vision = "<|vision_start|><|image_pad|><|vision_end|>"
    system_text = (
        "You are evaluating a visual search episode.\n"
        f"Target category: {target}\n"
        "You will be shown the sequence of fovea patches visited.\n"
        f"Reply with exactly {FINAL_FOUND} if the target is present, otherwise {FINAL_NOT_FOUND}.\n"
        "Do not output coordinates."
    )

    parts: list[str] = []
    parts.append(f"<|im_start|>system\n{system_text}<|im_end|>\n")
    for i in range(int(n_patches)):
        parts.append(f"<|im_start|>user\n{vision}\npatch_index={i}<|im_end|>\n")
    parts.append(
        f"<|im_start|>user\nAnswer with exactly {FINAL_FOUND} or {FINAL_NOT_FOUND}.<|im_end|>\n"
    )
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def _iter_chunks(idxs: np.ndarray, batch_size: int):
    n = int(len(idxs))
    bs = max(1, int(batch_size))
    for s in range(0, n, bs):
        yield idxs[s : s + bs]


def main() -> None:
    args = _parse_args()

    # vLLM uses multiprocessing; on Linux the default "fork" can crash if CUDA
    # was initialized before vLLM starts. "spawn" is slower to start but reliable.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "vLLM is not installed/working in this environment.\n"
            "Try installing it in your uv env (GPU wheel), then re-run.\n"
            f"Original import error: {e}"
        ) from e

    ds = COCOSearch18Dataset(
        args.data_dir,
        split=args.split,
        condition=args.condition,
        patch_size=tuple(args.patch_size),
    )

    idxs = np.arange(len(ds), dtype=np.int64)
    if args.shuffle:
        rng = np.random.default_rng(int(args.seed))
        rng.shuffle(idxs)
    if args.max_examples is not None:
        idxs = idxs[: int(args.max_examples)]

    enable_lora = args.lora_dir is not None
    llm = LLM(
        model=str(args.base_model_id),
        dtype="bfloat16",
        max_model_len=int(args.max_model_len),
        gpu_memory_utilization=float(args.gpu_mem_util),
        disable_log_stats=True,
        enforce_eager=True,
        limit_mm_per_prompt={"image": int(args.max_fixations)},
        enable_lora=bool(enable_lora),
        max_lora_rank=int(args.max_lora_rank),
    )
    lora_req = (
        LoRARequest("gaze", 1, str(Path(args.lora_dir)))
        if enable_lora
        else None
    )

    sampling = SamplingParams(temperature=0.0, max_tokens=8)

    tp = fp = tn = fn = 0
    n_present = n_absent = 0
    invalid_present = invalid_absent = 0
    pred_counts: Counter[str] = Counter()

    out_f = None
    if args.out_jsonl is not None:
        args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        out_f = open(args.out_jsonl, "w")

    try:
        for j, batch_idxs in enumerate(_iter_chunks(idxs, int(args.batch_size))):
            reqs = []
            metas = []

            for idx in batch_idxs.tolist():
                item = ds[int(idx)]
                target = str(item["target"])
                true_present = bool(item["condition"] == "present")

                if true_present:
                    n_present += 1
                else:
                    n_absent += 1

                T = int(item["patches"].shape[0])
                k = min(int(args.max_fixations), T)
                patch_imgs = [patch_tensor_to_pil(item["patches"][t]) for t in range(k)]
                prompt = _build_decision_prompt(target=target, n_patches=len(patch_imgs))

                reqs.append({"prompt": prompt, "multi_modal_data": {"image": patch_imgs}})
                metas.append(
                    {
                        "idx": int(idx),
                        "target": target,
                        "true_present": bool(true_present),
                    }
                )

            outs = llm.generate(
                reqs,
                sampling,
                use_tqdm=False,
                lora_request=lora_req,
            )

            for out, meta in zip(outs, metas):
                text = (out.outputs[0].text or "").strip()
                kind, val = _parse_reply(text)
                pred_final = str(val) if kind == "final" else None
                pred_counts[str(kind)] += 1

                true_present = bool(meta["true_present"])
                finalized = pred_final in {FINAL_FOUND, FINAL_NOT_FOUND}
                if finalized:
                    pred_present = bool(pred_final == FINAL_FOUND)
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
                        "idx": int(meta["idx"]),
                        "target": str(meta["target"]),
                        "true_condition": "present" if true_present else "absent",
                        "pred_present": bool(pred_present),
                        "raw_text": text,
                        "parsed_kind": str(kind),
                        "parsed_final": pred_final,
                    }
                    out_f.write(json.dumps(row) + "\n")

            done = min(int((j + 1) * int(args.batch_size)), int(len(idxs)))
            if done and (done % 256 == 0):
                acc = _safe_div(tp + tn, done)
                print(f"[{done}/{len(idxs)}] acc={acc:.3f} invalid={(invalid_present+invalid_absent)/done:.3f}")
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

    majority_acc = max(p_present, p_absent)
    uniform_random_acc = 0.5 if (n_present and n_absent) else 1.0
    prior_random_acc = (p_present * p_present) + (p_absent * p_absent)

    summary = {
        "backend": "vllm",
        "base_model_id": str(args.base_model_id),
        "lora_dir": str(args.lora_dir) if args.lora_dir is not None else None,
        "split": str(args.split),
        "condition": str(args.condition),
        "n_total": n_total,
        "n_present": int(n_present),
        "n_absent": int(n_absent),
        "present_rate": float(p_present),
        "invalid_present": int(invalid_present),
        "invalid_absent": int(invalid_absent),
        "parse_kinds": dict(pred_counts),
        "confusion": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
        "metrics": {
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "present_recall": float(acc_present),
            "absent_recall": float(acc_absent),
        },
        "baselines": {
            "majority_accuracy": float(majority_acc),
            "uniform_random_accuracy": float(uniform_random_acc),
            "prior_random_accuracy": float(prior_random_acc),
        },
        "eval": {
            "max_fixations": int(args.max_fixations),
            "patch_size": [int(args.patch_size[0]), int(args.patch_size[1])],
            "batch_size": int(args.batch_size),
            "max_model_len": int(args.max_model_len),
            "gpu_mem_util": float(args.gpu_mem_util),
        },
        "seed": int(args.seed),
    }

    print(json.dumps(summary, indent=2))

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()

