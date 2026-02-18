#!/usr/bin/env python3
"""
Optuna hyperparameter sweep for Qwen3-VL gaze SFT (LoRA).

Uses `transformers.Trainer.hyperparameter_search(backend="optuna")`, which supports
Optuna pruning based on intermediate eval loss.

Typical run (fast + prunable):
  uv run -m saccade.sweep_optuna_qwen3vl_gaze --n-trials 30 --max-steps 200
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Subset
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
    set_seed,
)

from saccade.coco_search18_dataset import COCOSearch18Dataset
from saccade.train_sft_qwen3vl_gaze import Qwen3VLGazeCollator


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--data-dir", type=Path, default=Path("data/coco_search18"))
    p.add_argument("--model-id", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--output-dir", type=Path, default=Path("runs/optuna-qwen3vl-gaze"))
    p.add_argument("--condition", choices=["TP", "TA", "all"], default="all")

    p.add_argument("--patch-size", type=int, nargs=2, default=(224, 224), metavar=("H", "W"))
    p.add_argument("--max-fixations", type=int, default=20)
    p.add_argument("--boxed-final", action="store_true")

    # Keep sweep runs small; this is for *ranking* hyperparams, not final training.
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--logging-steps", type=int, default=50)
    p.add_argument("--max-train-samples", type=int, default=512)
    p.add_argument("--max-eval-samples", type=int, default=128)
    p.add_argument("--dataloader-num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)

    # Search space (defaults chosen to be reasonable for LoRA on this task).
    p.add_argument("--lr-min", type=float, default=5e-5)
    p.add_argument("--lr-max", type=float, default=5e-4)
    p.add_argument("--warmup-ratio-min", type=float, default=0.0)
    p.add_argument("--warmup-ratio-max", type=float, default=0.08)
    p.add_argument("--lora-r-choices", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256, 384])
    p.add_argument("--lora-alpha-choices", type=int, nargs="+", default=[16, 32, 64])
    p.add_argument("--lora-dropout-min", type=float, default=0.0)
    p.add_argument("--lora-dropout-max", type=float, default=0.1)
    p.add_argument("--batch-size-choices", type=int, nargs="+", default=[1, 2])
    p.add_argument(
        "--effective-batch-size",
        type=int,
        default=8,
        help="Global batch size target. We derive gradient_accumulation_steps = ceil(effective_bs / micro_bs).",
    )

    # Optuna
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--timeout", type=int, default=None, help="Seconds. Optuna will stop after this wall time.")
    p.add_argument("--study-name", type=str, default="qwen3vl_gaze")
    p.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL, e.g. sqlite:////abs/path/to/study.db. If unset, uses output-dir/study.db",
    )
    p.add_argument("--resume", action="store_true", help="Resume an existing study (load_if_exists=True).")

    return p.parse_args()


def _subset(
    base: COCOSearch18Dataset, *, max_fixations: int, max_samples: int | None, seed: int
) -> Subset:
    idx = [
        i
        for i, sp in enumerate(base.trials)
        if 2 <= int(sp.get("length", 0)) <= int(max_fixations)
    ]
    rng = np.random.default_rng(int(seed))
    rng.shuffle(idx)
    if max_samples is not None:
        idx = idx[: int(max_samples)]
    return Subset(base, idx)


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Install a CUDA build of torch.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    storage = args.storage or f"sqlite:///{(args.output_dir / 'study.db').resolve()}"

    processor = AutoProcessor.from_pretrained(args.model_id)
    collator = Qwen3VLGazeCollator(processor=processor, boxed_final=args.boxed_final)

    train_base = COCOSearch18Dataset(
        args.data_dir,
        split="train",
        condition=args.condition,
        patch_size=tuple(args.patch_size),
    )
    eval_base = COCOSearch18Dataset(
        args.data_dir,
        split="valid",
        condition=args.condition,
        patch_size=tuple(args.patch_size),
    )

    train_ds = _subset(
        train_base,
        max_fixations=args.max_fixations,
        max_samples=args.max_train_samples,
        seed=args.seed,
    )
    eval_ds = _subset(
        eval_base,
        max_fixations=args.max_fixations,
        max_samples=args.max_eval_samples,
        seed=args.seed + 1,
    )

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    def model_init(trial=None):
        # trial is an optuna.Trial for sweeps; None for initial trainer construction.
        params = getattr(trial, "params", {}) or {}
        r = int(params.get("lora_r", 8))
        alpha = int(params.get("lora_alpha", 32))
        dropout = float(params.get("lora_dropout", 0.05))

        model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_id, torch_dtype=dtype)
        model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=lora_targets,
            bias="none",
        )
        return get_peft_model(model, lora_cfg)

    # Base training args; Optuna will override a subset each trial.
    base_warmup_steps = int(0.03 * args.max_steps)
    train_args = TrainingArguments(
        output_dir=str(args.output_dir),
        do_train=True,
        do_eval=True,
        max_steps=int(args.max_steps),
        learning_rate=2e-4,
        warmup_steps=base_warmup_steps,
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=int(math.ceil(args.effective_batch_size / 1)),
        logging_strategy="steps",
        logging_steps=int(args.logging_steps),
        eval_strategy="steps",
        eval_steps=int(args.eval_steps),
        save_strategy="no",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        dataloader_num_workers=int(args.dataloader_num_workers),
        remove_unused_columns=False,
        report_to=[],
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model_init=model_init,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        processing_class=processor,
    )

    def hp_space(trial):
        # Suggest both TrainingArguments params and model_init (LoRA) params.
        lr = float(trial.suggest_float("learning_rate", args.lr_min, args.lr_max, log=True))
        warmup_ratio = float(trial.suggest_float("warmup_ratio", args.warmup_ratio_min, args.warmup_ratio_max))

        # LoRA params: we only need to *suggest* them so they end up in trial.params.
        trial.suggest_categorical("lora_r", [int(x) for x in args.lora_r_choices])
        trial.suggest_categorical("lora_alpha", [int(x) for x in args.lora_alpha_choices])
        trial.suggest_float("lora_dropout", args.lora_dropout_min, args.lora_dropout_max)

        micro_bs = int(trial.suggest_categorical("per_device_train_batch_size", [int(x) for x in args.batch_size_choices]))
        grad_accum = int(math.ceil(float(args.effective_batch_size) / float(micro_bs)))
        warmup_steps = int(warmup_ratio * float(args.max_steps))

        return {
            "learning_rate": lr,
            "warmup_steps": warmup_steps,
            "per_device_train_batch_size": micro_bs,
            "gradient_accumulation_steps": grad_accum,
        }

    import optuna

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=args.eval_steps)

    best = trainer.hyperparameter_search(
        backend="optuna",
        direction="minimize",
        n_trials=int(args.n_trials),
        hp_space=hp_space,
        study_name=args.study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=bool(args.resume),
        n_jobs=1,
        timeout=args.timeout,
        gc_after_trial=True,
        catch=(torch.OutOfMemoryError,),
    )

    best_params = dict(best.hyperparameters)
    micro_bs = int(best_params.get("per_device_train_batch_size", 1))
    best_params["effective_batch_size"] = int(args.effective_batch_size)
    best_params["gradient_accumulation_steps"] = int(
        math.ceil(float(args.effective_batch_size) / float(micro_bs))
    )
    warmup_ratio = float(best_params.get("warmup_ratio", 0.0))
    best_params["warmup_steps"] = int(warmup_ratio * float(args.max_steps))

    out = {
        "best_run": {
            "run_id": best.run_id,
            "objective": best.objective,
            "params": best_params,
        }
    }
    (args.output_dir / "best.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
