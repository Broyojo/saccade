"""
Foveated Downscale
==================
Produces a genuinely smaller rectangular image where the center
gets more of the pixel budget and the edges are compressed.

The warp is SEPARABLE (applied independently to x and y), so
straight lines stay straight — no weird radial distortion.

The mapping uses sinh: f(t) = sinh(k*t)/sinh(k)
  - At center: sinh(x) ≈ x, so the mapping is ~linear (undistorted)
  - At edges: sinh grows fast, so many input pixels get crammed
    into few output pixels (lower effective resolution)

Usage:
    python foveated.py <image> [--size 300] [--k 2.0]
"""

import argparse
import os

import cv2
import numpy as np


def foveated_downscale(image, out_size=300, k=2.0, center=None):
    """
    Downscale an image with foveated (center-biased) sampling.

    Parameters
    ----------
    image    : (H, W, 3) input
    out_size : output size (square int) or (out_h, out_w)
    k        : foveation strength. 0 = uniform resize, higher = stronger.
               Typical range: 1.0 (mild) to 4.0 (aggressive)
    center   : (cx, cy) fixation point in input coords, default = center
    """
    h_in, w_in = image.shape[:2]
    if isinstance(out_size, (tuple, list, np.ndarray)):
        if len(out_size) != 2:
            raise ValueError(f"out_size must be an int or (out_h, out_w), got {out_size!r}")
        out_h, out_w = int(out_size[0]), int(out_size[1])
    else:
        out_h = out_w = int(out_size)
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"out_size must be positive, got {(out_h, out_w)!r}")

    if center is None:
        center = (w_in / 2.0, h_in / 2.0)
    cx, cy = center

    def make_grid_1d(n_out, n_in, c):
        """
        Map n_out output pixels to positions in [0, n_in-1]
        with denser sampling near input coordinate c.
        """
        # Normalized output coords in [-1, 1]
        t = np.linspace(-1.0, 1.0, n_out)

        if k < 0.01:
            # No foveation — uniform
            warped = t
        else:
            # sinh warp: linear at origin, compresses edges
            warped = np.sinh(k * t) / np.sinh(k)

        # warped is in [-1, 1]. Map to input coordinates.
        # -1 → 0, 0 → c, +1 → n_in-1
        # We split into left half (maps [0, c]) and right half (maps [c, n_in-1])
        input_coords = np.empty_like(warped)
        left = warped <= 0
        right = ~left
        input_coords[left] = c + warped[left] * c  # [-1,0] → [0, c]
        input_coords[right] = c + warped[right] * (n_in - 1 - c)  # [0,1] → [c, n_in-1]

        return input_coords.astype(np.float32)

    # Build 2D sample grid (separable = independent x and y)
    grid_x = make_grid_1d(out_w, w_in, cx)  # shape (out_w,)
    grid_y = make_grid_1d(out_h, h_in, cy)  # shape (out_h,)

    map_x, map_y = np.meshgrid(grid_x, grid_y)  # (out_h, out_w)

    out = cv2.remap(
        image, map_x, map_y, interpolation=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT
    )
    return out


def main():
    parser = argparse.ArgumentParser(description="Foveated Downscale")
    parser.add_argument("image", help="Input image")
    parser.add_argument("--size", type=int, default=300, help="Output size (square)")
    parser.add_argument(
        "--k",
        type=float,
        default=2.0,
        help="Foveation strength (0=uniform, 2=moderate, 4=strong)",
    )
    parser.add_argument("--cx", type=int, default=None)
    parser.add_argument("--cy", type=int, default=None)
    parser.add_argument("-o", "--output", default="foveated_output.png")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Cannot read {args.image}")

    h, w = img.shape[:2]
    center = None
    if args.cx is not None and args.cy is not None:
        center = (args.cx, args.cy)

    result = foveated_downscale(img, args.size, args.k, center)
    normal = cv2.resize(img, (args.size, args.size), interpolation=cv2.INTER_AREA)

    cv2.imwrite(args.output, result)
    base, ext = os.path.splitext(args.output)
    cv2.imwrite(f"{base}_normal{ext}", normal)

    print(f"Input:       {w} x {h}  ({w * h:,} px)")
    print(f"Output:      {args.size} x {args.size}  ({args.size**2:,} px)")
    print(f"Foveation k: {args.k}")
    print(f"Compression: {w * h / args.size**2:.1f}x")


if __name__ == "__main__":
    main()
