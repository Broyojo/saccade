"""
Foveated Quadtree Patches
==========================
Recursively subdivides the image into non-overlapping cells.
Cells near the fixation point are split finer (more detail).
Cells far away stay large (less detail).
Every cell is resized to the same patch_size → one token.

Zero overlap by construction.
"""

import argparse

import cv2
import numpy as np


def foveated_quadtree(
    image, center=None, patch_size=64, min_cell=64, max_cell=512, max_depth=6
):
    """
    Parameters
    ----------
    image     : (H, W, 3)
    center    : (cx, cy) fixation point
    patch_size: output token size (each cell resized to this)
    min_cell  : smallest cell size (at fovea)
    max_cell  : largest cell size (at periphery)
    max_depth : max recursion depth

    Returns
    -------
    patches : list of (patch_size, patch_size, 3)
    meta    : list of dicts {x1, y1, x2, y2, cell_size}
    """
    h, w = image.shape[:2]
    if center is None:
        center = (w / 2.0, h / 2.0)
    cx, cy = center

    max_dist = max(cx, cy, w - cx, h - cy)

    patches = []
    meta = []

    def should_split(x1, y1, x2, y2, depth):
        cell_w = x2 - x1
        cell_h = y2 - y1

        if cell_w <= min_cell or cell_h <= min_cell:
            return False

        if depth >= max_depth:
            return False

        # Distance from center of this cell to fixation point
        cell_cx = (x1 + x2) / 2
        cell_cy = (y1 + y2) / 2
        dist = np.sqrt((cell_cx - cx) ** 2 + (cell_cy - cy) ** 2)

        # Steeper falloff: use squared normalized distance
        t = min(1.0, (dist / max_dist) ** 0.7)
        threshold = min_cell + (max_cell - min_cell) * t

        return max(cell_w, cell_h) > threshold

    def subdivide(x1, y1, x2, y2, depth=0):
        if should_split(x1, y1, x2, y2, depth):
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            subdivide(x1, y1, mid_x, mid_y, depth + 1)  # top-left
            subdivide(mid_x, y1, x2, mid_y, depth + 1)  # top-right
            subdivide(x1, mid_y, mid_x, y2, depth + 1)  # bottom-left
            subdivide(mid_x, mid_y, x2, y2, depth + 1)  # bottom-right
        else:
            # Leaf node → extract patch
            crop = image[y1:y2, x1:x2]
            if crop.shape[0] < 2 or crop.shape[1] < 2:
                return
            token = cv2.resize(
                crop, (patch_size, patch_size), interpolation=cv2.INTER_AREA
            )
            patches.append(token)
            meta.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "cell_w": x2 - x1,
                    "cell_h": y2 - y1,
                }
            )

    subdivide(0, 0, w, h)
    return patches, meta


def visualize(image, patches, meta, patch_size=64):
    h, w = image.shape[:2]

    # --- Source with cell boundaries ---
    vis_src = image.copy()

    # Color by cell size
    sizes = [m["cell_w"] * m["cell_h"] for m in meta]
    min_s, max_s = min(sizes), max(sizes)

    for m in meta:
        s = m["cell_w"] * m["cell_h"]
        if max_s > min_s:
            t = (s - min_s) / (max_s - min_s)  # 0=small, 1=large
        else:
            t = 0
        # Green (small/fovea) → Red (large/periphery)
        color = (0, int(255 * (1 - t)), int(255 * t))
        cv2.rectangle(vis_src, (m["x1"], m["y1"]), (m["x2"], m["y2"]), color, 2)

    # --- Token grid ---
    n = len(patches)
    cols = min(n, 10)
    rows = (n + cols - 1) // cols
    gap = 2
    grid_w = cols * (patch_size + gap) + gap
    grid_h = rows * (patch_size + gap) + gap
    vis_tokens = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 40

    for i, (patch, m) in enumerate(zip(patches, meta)):
        r, c = divmod(i, cols)
        y = gap + r * (patch_size + gap)
        x = gap + c * (patch_size + gap)
        vis_tokens[y : y + patch_size, x : x + patch_size] = patch

        s = m["cell_w"] * m["cell_h"]
        if max_s > min_s:
            t = (s - min_s) / (max_s - min_s)
        else:
            t = 0
        color = (0, int(255 * (1 - t)), int(255 * t))
        cv2.rectangle(
            vis_tokens, (x - 1, y - 1), (x + patch_size, y + patch_size), color, 1
        )

    return vis_src, vis_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--cx", type=int, default=None)
    parser.add_argument("--cy", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--min-cell", type=int, default=64)
    parser.add_argument("--max-cell", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("-o", "--output", default="quadtree")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    h, w = img.shape[:2]
    center = (args.cx or w // 2, args.cy or h // 2)

    patches, meta = foveated_quadtree(
        img,
        center,
        args.patch_size,
        args.min_cell,
        args.max_cell,
        args.max_depth,
    )

    vis_src, vis_tokens = visualize(img, patches, meta, args.patch_size)
    cv2.imwrite(f"{args.output}_source.png", vis_src)
    cv2.imwrite(f"{args.output}_tokens.png", vis_tokens)

    print(f"Image:       {w} x {h} ({w * h:,} px)")
    print(f"Tokens:      {len(patches)} patches of {args.patch_size}x{args.patch_size}")
    print(f"Token pixels: {len(patches) * args.patch_size**2:,}")
    print(f"Compression: {w * h / (len(patches) * args.patch_size**2):.1f}x")

    # Stats by cell size
    from collections import Counter

    size_counts = Counter(m["cell_w"] for m in meta)
    for sz in sorted(size_counts):
        n = size_counts[sz]
        ds = sz / args.patch_size
        print(f"  {sz:4d}px cells: {n:3d} tokens (effective {ds:.1f}x downsample)")


if __name__ == "__main__":
    main()
