#!/usr/bin/env python3
"""
Roll out a trained Qwen3-VL gaze policy (LoRA) on a single image.

Given a target category, the model iteratively predicts fixations:
  user:      patch at current fixation
  assistant: "x y" (next fixation) or FOUND/NOT_FOUND (stop)

The script saves a visualization similar to `visualize_coco_search18_example.py`,
but using the *model-predicted* trajectory.

Examples
  # Rollout on a dataset example (uses the trial's image + target):
  uv run -m saccade.rollout_qwen3vl_gaze --idx 0 --out-dir vis/rollouts --no-show

  # Rollout on an arbitrary image:
  uv run -m saccade.rollout_qwen3vl_gaze --image-path path/to/img.jpg --target "cup" --out-dir vis/rollouts --no-show
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, set_seed

from saccade.coco_search18_dataset import ORIG_H, ORIG_W, COCOSearch18Dataset
from saccade.model import FINAL_FOUND, FINAL_NOT_FOUND, encode_messages


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Model/checkpoint.
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
    p.add_argument(
        "--merge-lora",
        action="store_true",
        help="Merge LoRA into the base model for faster inference.",
    )

    # Input selection.
    p.add_argument(
        "--image-path", type=Path, default=None, help="Path to an RGB image."
    )
    p.add_argument(
        "--target", type=str, default=None, help="Target category string (e.g., 'cup')."
    )

    p.add_argument("--data-dir", type=Path, default=Path("data/coco_search18"))
    p.add_argument("--split", default="valid", choices=["train", "valid", "all"])
    p.add_argument("--condition", default="TP", choices=["TP", "TA", "all"])
    p.add_argument(
        "--idx",
        type=int,
        default=None,
        help="If set, run on this COCO-Search18 dataset index.",
    )

    # Rollout params.
    p.add_argument(
        "--patch-size", type=int, nargs=2, default=(224, 224), metavar=("H", "W")
    )
    p.add_argument(
        "--max-fixations",
        type=int,
        default=12,
        help="Maximum number of fixations to visit (including start).",
    )
    p.add_argument(
        "--start-xy",
        type=int,
        nargs=2,
        default=None,
        metavar=("X0", "Y0"),
        help="Starting fixation (x y).",
    )
    p.add_argument(
        "--no-resize",
        action="store_true",
        help=f"Do not resize to {ORIG_W}x{ORIG_H}. (Not recommended; model was trained in that coordinate space.)",
    )

    # Generation params.
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument(
        "--sample",
        action="store_true",
        help="Enable sampling (otherwise greedy decoding).",
    )
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)

    # Output/vis.
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("vis/rollouts"),
        help="Write PNG + JSON outputs here.",
    )
    p.add_argument(
        "--boxes", action="store_true", help="Draw patch boxes on the full image."
    )
    p.add_argument(
        "--px-per-deg",
        type=float,
        default=32.0,
        help="For amplitude plot: pixels per degree.",
    )
    p.add_argument(
        "--no-show", action="store_true", help="Do not open interactive windows."
    )

    # Misc.
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")

    return p.parse_args()


def _read_base_model_id(model_dir: Path) -> str | None:
    cfg_path = model_dir / "adapter_config.json"
    if not cfg_path.exists():
        return None
    with open(cfg_path) as f:
        cfg = json.load(f)
    v = cfg.get("base_model_name_or_path")
    return str(v) if v else None


def _default_system_prompt(target: str) -> str:
    return (
        "You are a gaze policy for visual search.\n"
        f"Target category: {target}\n"
        "On each turn, reply with either:\n"
        '  - the next fixation as two integers: "x y"\n'
        f"  - or {FINAL_FOUND}/{FINAL_NOT_FOUND} if you want to stop.\n"
        "Do not output anything else."
    )


def _crop_patch_rgb(
    full_rgb: Image.Image, *, cx: int, cy: int, patch_size: tuple[int, int]
) -> Image.Image:
    # Reflect-pad then crop, matching the dataset's behavior.
    pH, pW = patch_size
    pad_h, pad_w = pH // 2 + 1, pW // 2 + 1

    arr = np.asarray(full_rgb)  # [H,W,3] uint8
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB image array [H,W,3], got shape={arr.shape}")

    arr_p = np.pad(arr, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="reflect")
    cx_p = int(round(cx)) + pad_w
    cy_p = int(round(cy)) + pad_h
    top = cy_p - (pH // 2)
    left = cx_p - (pW // 2)
    patch = arr_p[top : top + pH, left : left + pW]
    return Image.fromarray(patch, mode="RGB")


_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _parse_reply(text: str) -> tuple[str, Any]:
    """
    Returns:
      ("final", "FOUND"|"NOT_FOUND")
      ("xy", (x:int, y:int))
      ("error", reason:str)
    """
    s = (text or "").strip()
    if not s:
        return ("error", "empty")

    # Prefer NOT_FOUND over FOUND since it contains the substring "FOUND".
    if "NOT_FOUND" in s:
        return ("final", FINAL_NOT_FOUND)
    if "FOUND" in s:
        return ("final", FINAL_FOUND)

    # Handle boxed variants like \boxed{FOUND}.
    m = re.search(r"\\boxed\{(FOUND|NOT_FOUND)\}", s)
    if m:
        return ("final", m.group(1))

    nums = _NUM_RE.findall(s)
    if len(nums) < 2:
        return ("error", f"could_not_parse_xy: {s!r}")

    try:
        x = int(round(float(nums[0])))
        y = int(round(float(nums[1])))
    except ValueError:
        return ("error", f"bad_number: {nums[:2]!r}")

    return ("xy", (x, y))


def _plot_rollout(
    *,
    full_rgb: Image.Image,
    fixations: list[tuple[int, int]],
    patches: list[Image.Image],
    patch_size: tuple[int, int],
    amps_deg: np.ndarray,
    title: str,
    boxes: bool,
    out_path: Path | None,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Rectangle

    full = np.asarray(full_rgb)
    xs = np.asarray([x for x, _ in fixations], dtype=np.float32)
    ys = np.asarray([y for _, y in fixations], dtype=np.float32)
    nfix = int(len(fixations))

    fig = plt.figure(figsize=(max(12.0, 1.9 * nfix), 8.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[3.0, 1.2], width_ratios=[4.0, 1.3])
    ax_img = fig.add_subplot(gs[0, 0])
    ax_amp = fig.add_subplot(gs[0, 1])

    ax_img.imshow(full)
    ax_img.set_title(title, fontsize=10)
    ax_img.set_axis_off()

    if nfix >= 2:
        segs = np.stack(
            [np.stack([xs[:-1], ys[:-1]], axis=1), np.stack([xs[1:], ys[1:]], axis=1)],
            axis=1,
        )  # [nfix-1, 2, 2]
        lc = LineCollection(segs, cmap="viridis", linewidths=2.0, zorder=2)
        lc.set_array(amps_deg.astype(np.float32))
        ax_img.add_collection(lc)
        cbar = fig.colorbar(lc, ax=ax_img, fraction=0.046, pad=0.01)
        cbar.set_label("saccade amplitude (deg)")

        ax_amp.plot(np.arange(len(amps_deg)), amps_deg, marker="o", linewidth=1.5)
        ax_amp.set_title("Saccade amplitude", fontsize=10)
        ax_amp.set_xlabel("saccade index")
        ax_amp.set_ylabel("deg")
        ax_amp.grid(True, alpha=0.3)
    else:
        ax_amp.set_axis_off()

    ax_img.scatter(
        xs,
        ys,
        s=30,
        c=np.arange(nfix),
        cmap="plasma",
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
    )
    ax_img.scatter(
        [xs[0]], [ys[0]], s=80, c="lime", edgecolors="black", linewidths=0.8, zorder=4
    )
    ax_img.scatter(
        [xs[-1]], [ys[-1]], s=80, c="red", edgecolors="black", linewidths=0.8, zorder=4
    )

    for i, (x, y) in enumerate(zip(xs, ys)):
        ax_img.text(
            float(x),
            float(y),
            str(i),
            color="white",
            fontsize=8,
            ha="center",
            va="center",
            zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.6),
        )

    if boxes:
        pH, pW = patch_size
        for x, y in zip(xs, ys):
            ax_img.add_patch(
                Rectangle(
                    (float(x) - pW / 2.0, float(y) - pH / 2.0),
                    float(pW),
                    float(pH),
                    fill=False,
                    edgecolor="white",
                    linewidth=1.0,
                    alpha=0.35,
                    zorder=1,
                )
            )

    patch_gs = gs[1, :].subgridspec(1, nfix, wspace=0.05)
    axes = [fig.add_subplot(patch_gs[0, i]) for i in range(nfix)]
    for i, ax in enumerate(axes):
        ax.imshow(np.asarray(patches[i]))
        ax.set_axis_off()
        if nfix >= 2 and i < nfix - 1:
            ax.set_title(f"fix {i}  amp→ {float(amps_deg[i]):.2f}°", fontsize=9)
        else:
            ax.set_title(f"fix {i}", fontsize=9)
        H, W = patches[i].size[1], patches[i].size[0]
        ax.scatter([W / 2.0], [H / 2.0], s=16, c="red", marker="+")

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"Saved: {out_path}")

    if show:
        plt.show()

    plt.close(fig)


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)

    if args.max_fixations < 1:
        raise ValueError("--max-fixations must be >= 1")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Install a CUDA build of torch.")

    # Select image + target.
    ds_item = None
    ds_sp = None
    if args.idx is not None:
        ds = COCOSearch18Dataset(
            args.data_dir,
            split=args.split,
            condition=args.condition,
            patch_size=tuple(args.patch_size),
        )
        ds_sp = ds.trials[int(args.idx)]
        ds_item = ds[int(args.idx)]
        image_path = ds._get_image_path(ds_sp)
        target = str(args.target or ds_item["target"])
        if args.start_xy is None:
            x0, y0 = map(int, ds_item["fixation_xy"][0].tolist())
        else:
            x0, y0 = map(int, args.start_xy)
        gt_found = bool(ds_item["found"])
        gt_condition = str(ds_item["condition"])
    else:
        if args.image_path is None or args.target is None:
            raise ValueError(
                "Provide either --idx (dataset mode) or both --image-path and --target."
            )
        image_path = Path(args.image_path)
        target = str(args.target)
        if args.start_xy is None:
            x0, y0 = ORIG_W // 2, ORIG_H // 2
        else:
            x0, y0 = map(int, args.start_xy)
        gt_found = None
        gt_condition = None

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    full_rgb = Image.open(image_path).convert("RGB")
    orig_size = tuple(full_rgb.size)
    if not args.no_resize:
        full_rgb = full_rgb.resize((ORIG_W, ORIG_H), resample=Image.BICUBIC)
    W, H = full_rgb.size

    # Load model/processor.
    model_dir = Path(args.model_dir)
    base_model_id = (
        args.base_model_id
        or _read_base_model_id(model_dir)
        or "Qwen/Qwen3-VL-2B-Instruct"
    )

    device = torch.device(args.device)
    dtype = (
        torch.bfloat16
        if (device.type == "cuda" and torch.cuda.is_bf16_supported())
        else torch.float16
    )

    processor = AutoProcessor.from_pretrained(str(model_dir))
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_id, torch_dtype=dtype
    )
    model = PeftModel.from_pretrained(base, str(model_dir))
    if args.merge_lora and hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()
    model.to(device)
    model.eval()

    # Rollout.
    fixations: list[tuple[int, int]] = [(int(x0), int(y0))]
    patches: list[Image.Image] = []
    outputs: list[str] = []
    stop_reason = "max_fixations"
    final_text = None
    suggested_next_xy = None

    system_prompt = _default_system_prompt(target)

    for _t in range(int(args.max_fixations)):
        cx, cy = fixations[-1]
        cx = int(max(0, min(cx, W - 1)))
        cy = int(max(0, min(cy, H - 1)))

        patch = _crop_patch_rgb(
            full_rgb, cx=cx, cy=cy, patch_size=tuple(args.patch_size)
        )
        patches.append(patch)

        # Build messages: (system) + previous (user patch, assistant text) pairs + current user patch.
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
        ]
        for p_img, out_text in zip(patches[:-1], outputs):
            messages.append(
                {"role": "user", "content": [{"type": "image", "image": p_img}]}
            )
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": out_text}]}
            )
        messages.append(
            {"role": "user", "content": [{"type": "image", "image": patches[-1]}]}
        )

        batch = encode_messages(
            processor, messages, add_generation_prompt=True, device=device
        )
        in_len = int(batch.inputs["input_ids"].shape[1])

        gen_kwargs = {
            "max_new_tokens": int(args.max_new_tokens),
            "do_sample": bool(args.sample),
            "pad_token_id": processor.tokenizer.pad_token_id,
            "eos_token_id": processor.tokenizer.eos_token_id,
        }
        if args.sample:
            gen_kwargs["temperature"] = float(args.temperature)
            gen_kwargs["top_p"] = float(args.top_p)

        with (
            torch.inference_mode(),
            torch.autocast(device_type=device.type, dtype=dtype),
        ):
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
            if len(fixations) >= int(args.max_fixations):
                suggested_next_xy = (x1, y1)
                stop_reason = "max_fixations"
                break
            fixations.append((x1, y1))
            continue

        stop_reason = "parse_error"
        final_text = None
        break

    # Compute amplitudes (deg) for visualization.
    if len(fixations) >= 2:
        xs = np.asarray([x for x, _ in fixations], dtype=np.float32)
        ys = np.asarray([y for _, y in fixations], dtype=np.float32)
        dist_px = np.hypot(xs[1:] - xs[:-1], ys[1:] - ys[:-1])
        amps_deg = dist_px / float(args.px_per_deg)
    else:
        amps_deg = np.zeros((0,), dtype=np.float32)

    # Save JSON + PNG.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem_bits = [
        f"idx{args.idx}" if args.idx is not None else Path(image_path).stem,
        re.sub(r"[^a-zA-Z0-9]+", "_", target).strip("_") or "target",
        stop_reason,
    ]
    stem = "_".join(stem_bits)
    png_path = args.out_dir / f"{stem}_rollout.png"
    json_path = args.out_dir / f"{stem}_rollout.json"

    payload = {
        "model_dir": str(model_dir),
        "base_model_id": str(base_model_id),
        "image_path": str(Path(image_path)),
        "target": target,
        "orig_image_size": list(orig_size),
        "resized_image_size": list(full_rgb.size),
        "patch_size": [int(args.patch_size[0]), int(args.patch_size[1])],
        "start_xy": [int(x0), int(y0)],
        "fixation_xy": [[int(x), int(y)] for x, y in fixations],
        "assistant_outputs": outputs,
        "stop_reason": stop_reason,
        "final_text": final_text,
        "suggested_next_xy": list(suggested_next_xy)
        if suggested_next_xy is not None
        else None,
        "seed": int(args.seed),
        "device": str(args.device),
        "dtype": str(dtype).replace("torch.", ""),
        "gt_found": gt_found,
        "gt_condition": gt_condition,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {json_path}")

    title = f"target={target}  stop={stop_reason}  final={final_text}  nfix={len(fixations)}"
    if args.idx is not None:
        title = f"idx={args.idx}  {Path(image_path).name}  {title}  gt_found={gt_found}  gt_cond={gt_condition}"

    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")

    _plot_rollout(
        full_rgb=full_rgb,
        fixations=fixations,
        patches=patches,
        patch_size=tuple(args.patch_size),
        amps_deg=amps_deg,
        title=title,
        boxes=bool(args.boxes),
        out_path=png_path,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
