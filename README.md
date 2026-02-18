# saccade

## Download COCO-Search18
```bash
bash download_coco_search18.sh
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
