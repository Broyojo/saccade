#!/usr/bin/env bash

uv run -m saccade.train_sft_qwen3vl_gaze \
  --condition all \
  --patch-size 224 224 \
  --max-fixations 20 \
  --lora-r 256 --lora-alpha 32 --lora-dropout 0.05 \
  --learning-rate 2e-4 --warmup-ratio 0.03 --weight-decay 0.0 \
  --per-device-train-batch-size 1 --gradient-accumulation-steps 8 \
  --num-train-epochs 1 \
  --max-eval-samples 512 --eval-steps 1000 --save-steps 1000 \
  --eval-first --wandb-project saccade
