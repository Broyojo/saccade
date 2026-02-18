# saccade

## Download COCO-Search18
```bash
bash scripts/download_coco_search18.sh
```

## Torch Dataset
Each item is one trial (full scanpath) with:
- `patches`: `[T, 3, pH, pW]`
- `fixation_xy`: `[T, 2]` (int, original 1680x1050 coords)
- `saccade_amplitude`: `[T-1]`
- `target`: `str`
- `found`: `bool`
- `condition`: `"present"` or `"absent"`

Quick sanity check:
```bash
uv run -m saccade.coco_search18_dataset
```

## Visualize One Trial
```bash
uv run -m saccade.visualize_coco_search18_example --idx 0 --max-fix 12 --boxes
```

Headless (save PNGs):
```bash
uv run -m saccade.visualize_coco_search18_example --idx 0 --max-fix 12 --out-dir ./vis --no-show
```

## Roll Out The Trained Policy (LLM Trajectory)
This runs the trained Qwen3-VL+LoRA policy in a closed loop: it predicts the next fixation from
the current patch until it emits `FOUND`/`NOT_FOUND` (or hits `--max-fixations`), then saves a PNG
trajectory visualization and a JSON log.

Dataset mode (pick an index, uses that image + target):
```bash
uv run -m saccade.rollout_qwen3vl_gaze --idx 0 --out-dir ./vis/rollouts --no-show --boxes
```

Arbitrary image mode:
```bash
uv run -m saccade.rollout_qwen3vl_gaze --image-path path/to/img.jpg --target "cup" --out-dir ./vis/rollouts --no-show --boxes
```

## Train (Qwen3-VL + LoRA)
Weights/logs are written under `runs/` by default.

Fit check (runs 1 train step + 1 eval step on the longest trials under `--max-fixations`):
```bash
uv run -m saccade.bench_fit_qwen3vl_gaze --max-fixations 48 --lora-r 256
```

W&B:
```bash
uv run wandb login
```

Quick smoke run (no W&B, tiny subset):
```bash
uv run -m saccade.train_sft_qwen3vl_gaze --no-wandb --max-train-samples 8 --max-eval-samples 8 --eval-steps 5 --save-steps 5
```

Real run (uses W&B):
```bash
uv run -m saccade.train_sft_qwen3vl_gaze --wandb-project saccade
```

## Eval (Present/Absent)
Roll out the model on TP+TA and score whether it predicts `FOUND` (present) vs `NOT_FOUND` (absent).

```bash
uv run -m saccade.eval_rollout_qwen3vl_gaze --split valid --condition all --max-fixations 12 --force-final
```

## Hyperparameter Sweep (Optuna)
This runs small, prunable training runs and optimizes `eval_loss`.

```bash
uv run -m saccade.sweep_optuna_qwen3vl_gaze --n-trials 30 --max-steps 200 --eval-steps 50 --resume
```

Or:
```bash
bash scripts/sweep_optuna.sh
```
