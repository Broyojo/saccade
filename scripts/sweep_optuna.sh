#!/usr/bin/env bash
set -euo pipefail

# Small Optuna sweep (prunable) to tune LoRA + optimizer hyperparams.
uv run -m saccade.sweep_optuna_qwen3vl_gaze \
  --n-trials 30 \
  --max-steps 200 \
  --eval-steps 50 \
  --max-train-samples 512 \
  --max-eval-samples 128 \
  --resume

