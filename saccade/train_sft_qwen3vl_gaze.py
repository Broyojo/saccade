#!/usr/bin/env python3
"""
SFT (behavior cloning) for a gaze policy using Qwen3-VL + LoRA.

We train on full trial rollouts as an interleaved chat:
  user:      patch image at fixation t
  assistant: "x_{t+1} y_{t+1}"  (for t < T-1)
  assistant: FOUND/NOT_FOUND    (for t == T-1)

This script is intentionally minimal and uses Transformers Trainer + PEFT.
Run:
  uv run -m saccade.train_sft_qwen3vl_gaze --help
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoProcessor, Trainer, TrainingArguments, set_seed

from saccade.coco_search18_dataset import COCOSearch18Dataset
from saccade.model import build_rollout_messages, patch_tensor_to_pil


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--data-dir", type=Path, default=Path("data/coco_search18"))
    p.add_argument("--model-id", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--output-dir", type=Path, default=Path("runs/qwen3vl-gaze-lora"))
    p.add_argument("--condition", choices=["TP", "TA", "all"], default="all")

    p.add_argument(
        "--patch-size", type=int, nargs=2, default=(224, 224), metavar=("H", "W")
    )
    p.add_argument(
        "--max-fixations", type=int, default=12, help="Skip trials longer than this."
    )
    p.add_argument(
        "--boxed-final",
        action="store_true",
        help=r"Use \\boxed{FOUND} style final answer.",
    )

    p.add_argument("--num-train-epochs", type=float, default=1.0)
    # Many LoRA setups prefer LR ~10x full fine-tuning (see Thinking Machines LoRA study).
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.0)

    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--per-device-eval-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)

    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--save-total-limit", type=int, default=2)

    p.add_argument("--eval-first", action="store_true", help="Run eval once before training.")
    p.add_argument("--eval-only", action="store_true", help="Only run eval and exit.")

    p.add_argument("--dataloader-num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-eval-samples", type=int, default=None)

    p.add_argument("--wandb-project", type=str, default="saccade")
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--no-wandb", action="store_true")

    # LoRA
    p.add_argument("--lora-r", type=int, default=8)
    # Following standard PEFT practice + the Thinking Machines LoRA study.
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--lora-target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    return p.parse_args()


class RolloutDataset(Dataset):
    """
    Wraps COCOSearch18Dataset and filters out trials longer than max_fixations.

    Returns the raw trial item; the collator handles chat formatting + tokenization.
    """

    def __init__(
        self,
        base: COCOSearch18Dataset,
        *,
        max_fixations: int,
        max_samples: int | None = None,
    ):
        self.base = base
        self.indices = [
            i
            for i, sp in enumerate(base.trials)
            if 2 <= int(sp.get("length", 0)) <= int(max_fixations)
        ]
        if max_samples is not None:
            self.indices = self.indices[: int(max_samples)]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict[str, Any]:
        return self.base[self.indices[i]]


@dataclass
class Qwen3VLGazeCollator:
    processor: Any
    boxed_final: bool = False

    def __post_init__(self) -> None:
        tok = self.processor.tokenizer
        self.im_start_id = tok.convert_tokens_to_ids("<|im_start|>")
        self.im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
        self.assistant_id = tok.convert_tokens_to_ids("assistant")
        self.newline_id = tok.encode("\n", add_special_tokens=False)[0]

    def _assistant_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Mask assistant message tokens (content + <|im_end|>), for all assistant turns.

        We detect spans:
          <|im_start|> assistant \\n  ...  <|im_end|>
        and label everything from after the newline through the <|im_end|>.
        """
        ids = input_ids.tolist()
        L = len(ids)
        mask = [False] * L

        i = 0
        while i < L - 1:
            if ids[i] == self.im_start_id and ids[i + 1] == self.assistant_id:
                start = i + 2
                if start < L and ids[start] == self.newline_id:
                    start += 1
                j = start
                while j < L and ids[j] != self.im_end_id:
                    j += 1
                if j < L and ids[j] == self.im_end_id:
                    for k in range(start, j + 1):
                        mask[k] = True
                    i = j + 1
                    continue
                # If no <|im_end|>, label to end.
                for k in range(start, L):
                    mask[k] = True
                break
            i += 1

        return torch.tensor(mask, dtype=torch.bool)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts: list[str] = []
        flat_images = []

        for f in features:
            patches = f["patches"]
            fixation_xy = f["fixation_xy"]
            target = f["target"]
            found = bool(f["found"])

            patch_imgs = [patch_tensor_to_pil(p) for p in patches]
            xy = [tuple(map(int, v.tolist())) for v in fixation_xy]

            messages = build_rollout_messages(
                patch_imgs,
                xy,
                target=target,
                found=found,
                boxed_final=self.boxed_final,
            )
            text = self.processor.apply_chat_template(messages, tokenize=False)

            texts.append(text)
            flat_images.extend(patch_imgs)

        batch = self.processor(
            text=texts,
            images=flat_images,
            padding=True,
            return_tensors="pt",
        )

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = input_ids.clone()

        # Only train on assistant spans.
        keep = torch.stack([self._assistant_mask(row) for row in input_ids], dim=0)
        labels[~keep] = -100
        labels[attention_mask == 0] = -100
        batch["labels"] = labels
        return batch


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Install a CUDA build of torch.")

    if not args.no_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    processor = AutoProcessor.from_pretrained(args.model_id)

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

    train_ds = RolloutDataset(
        train_base,
        max_fixations=args.max_fixations,
        max_samples=args.max_train_samples,
    )
    eval_ds = RolloutDataset(
        eval_base,
        max_fixations=args.max_fixations,
        max_samples=args.max_eval_samples,
    )

    # Load model on CPU; accelerate/Trainer will place it on GPU.
    from transformers import Qwen3VLForConditionalGeneration

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=dtype
    )
    model.config.use_cache = False

    # Gradient checkpointing helps a lot with long multimodal sequences.
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
    model.print_trainable_parameters()

    report_to = [] if args.no_wandb else ["wandb"]

    # Transformers 5.2 deprecates warmup_ratio in favor of warmup_steps.
    steps_per_epoch = math.ceil(
        len(train_ds)
        / (args.per_device_train_batch_size * args.gradient_accumulation_steps)
    )
    total_train_steps = math.ceil(steps_per_epoch * float(args.num_train_epochs))
    warmup_steps = int(total_train_steps * float(args.warmup_ratio))

    train_args = TrainingArguments(
        output_dir=str(args.output_dir),
        do_train=True,
        do_eval=True,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        report_to=report_to,
        run_name=args.wandb_run_name,
        gradient_checkpointing=True,
    )

    collator = Qwen3VLGazeCollator(processor=processor, boxed_final=args.boxed_final)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        processing_class=processor,
    )

    if args.eval_first or args.eval_only:
        metrics = trainer.evaluate()
        # Make the baseline visible even with --no-wandb.
        print({"pretrain_" + k: v for k, v in metrics.items()})
        if args.eval_only:
            return

    trainer.train()
    trainer.save_model()
    processor.save_pretrained(str(args.output_dir))


if __name__ == "__main__":
    main()
