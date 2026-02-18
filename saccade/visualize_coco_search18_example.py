"""
Visualize one COCO-Search18 trial:
  - full image with fixation path (segments colored by saccade amplitude)
  - patch sequence (one patch per fixation) placed underneath the full image
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch

from saccade.coco_search18_dataset import COCOSearch18Dataset

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def _unnormalize(img: torch.Tensor) -> torch.Tensor:
    return (img * IMAGENET_STD + IMAGENET_MEAN).clamp(0.0, 1.0)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/coco_search18"))
    p.add_argument("--split", default="train", choices=["train", "valid", "all"])
    p.add_argument("--condition", default="TP", choices=["TP", "TA", "all"])
    p.add_argument(
        "--idx",
        type=int,
        default=None,
        help="Dataset index; default picks a random one",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--max-fix", type=int, default=12, help="Max fixations to visualize (prefix)"
    )
    p.add_argument(
        "--patch-size", type=int, nargs=2, default=(224, 224), metavar=("H", "W")
    )
    p.add_argument(
        "--foveated-patches",
        action="store_true",
        help="Apply separable foveated resampling to each patch (output stays patch-size).",
    )
    p.add_argument(
        "--foveated-k",
        type=float,
        default=2.0,
        help="Foveation strength for --foveated-patches (0=uniform).",
    )
    p.add_argument(
        "--boxes", action="store_true", help="Draw patch boxes on the full image"
    )
    p.add_argument("--out-dir", type=Path, default=None, help="If set, save PNGs here")
    p.add_argument(
        "--no-show", action="store_true", help="Do not open interactive windows"
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.out_dir is not None:
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Rectangle
    from PIL import Image

    ds = COCOSearch18Dataset(
        args.data_dir,
        split=args.split,
        condition=args.condition,
        patch_size=tuple(args.patch_size),
    )

    if args.idx is None:
        rng = random.Random(args.seed)
        idx = rng.randrange(len(ds))
    else:
        idx = args.idx
    sp = ds.trials[idx]
    item = ds[idx]

    xy = item["fixation_xy"].detach().cpu().numpy().astype(np.int32)
    xs = xy[:, 0].astype(np.float32)
    ys = xy[:, 1].astype(np.float32)
    T = int(xs.shape[0])
    nfix = max(2, min(int(args.max_fix), T))

    xs = xs[:nfix]
    ys = ys[:nfix]
    amps = (
        item["saccade_amplitude"][: nfix - 1].detach().cpu().numpy().astype(np.float32)
    )
    patches = item["patches"][:nfix]

    img_path = ds._get_image_path(sp)
    full = np.asarray(Image.open(img_path).convert("RGB"))

    target = item["target"]
    found = bool(item["found"])
    condition = item["condition"]
    subject = sp.get("subject", "?")
    split = sp.get("split", "?")

    title = (
        f"idx={idx}  {Path(img_path).name}  subj={subject}  split={split}  "
        f"condition={condition}  target={target}  found={found}  "
        f"T={T} (showing {nfix})"
    )

    # --- Combined figure: image/path on top, patch row on bottom ---
    fig = plt.figure(figsize=(max(12.0, 1.9 * nfix), 8.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[3.0, 1.2], width_ratios=[4.0, 1.3])
    ax_img = fig.add_subplot(gs[0, 0])
    ax_amp = fig.add_subplot(gs[0, 1])

    ax_img.imshow(full)
    ax_img.set_title(title, fontsize=10)
    ax_img.set_axis_off()

    # segments colored by amplitude
    segs = np.stack(
        [np.stack([xs[:-1], ys[:-1]], axis=1), np.stack([xs[1:], ys[1:]], axis=1)],
        axis=1,
    )  # [nfix-1, 2, 2]
    lc = LineCollection(segs, cmap="viridis", linewidths=2.0, zorder=2)
    lc.set_array(amps)
    ax_img.add_collection(lc)
    cbar = fig.colorbar(lc, ax=ax_img, fraction=0.046, pad=0.01)
    cbar.set_label("saccade amplitude (deg)")

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

    if args.boxes:
        pH, pW = args.patch_size
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

    ax_amp.plot(np.arange(len(amps)), amps, marker="o", linewidth=1.5)
    ax_amp.set_title("Saccade amplitude", fontsize=10)
    ax_amp.set_xlabel("saccade index")
    ax_amp.set_ylabel("deg")
    ax_amp.grid(True, alpha=0.3)

    n = nfix
    patch_gs = gs[1, :].subgridspec(1, n, wspace=0.05)
    axes = [fig.add_subplot(patch_gs[0, i]) for i in range(n)]
    for i, ax in enumerate(axes):
        p = _unnormalize(patches[i]).permute(1, 2, 0).detach().cpu().numpy()
        if args.foveated_patches:
            from saccade.separable_variable_density_foveated import foveated_downscale

            p8 = (p * 255.0).round().astype(np.uint8)
            p8 = foveated_downscale(
                p8, out_size=(int(p8.shape[0]), int(p8.shape[1])), k=float(args.foveated_k)
            )
            ax.imshow(p8)
        else:
            ax.imshow(p)
        ax.set_axis_off()

        if i < n - 1:
            ax.set_title(f"fix {i}  amp→ {float(amps[i]):.2f}°", fontsize=9)
        else:
            ax.set_title(f"fix {i} (last)", fontsize=9)

        # mark patch center (fixation is centered here)
        H, W = p.shape[:2]
        ax.scatter([W / 2.0], [H / 2.0], s=16, c="red", marker="+")

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"cocosearch18_idx{idx}_{condition}_{split}_subj{subject}_{target}"
        out = args.out_dir / f"{stem}_overview.png"
        fig.savefig(out, dpi=150)
        print(f"Saved: {out}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
