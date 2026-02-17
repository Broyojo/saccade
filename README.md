# saccade

## Download COCO-Search18
```bash
bash download_coco_search18.sh
```

## Torch Dataset
Each item is one trial (full scanpath) with:
- `patches`: `[T, 3, pH, pW]`
- `saccade_amplitude`: `[T-1]`
- `target`: `str`
- `found`: `bool`
- `condition`: `"present"` or `"absent"`

Quick sanity check:
```bash
uv run -m saccade.coco_search18_dataset
```
