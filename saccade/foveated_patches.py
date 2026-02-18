import cv2
import numpy as np


def sample_foveated_patches(
    image,
    center=None,
    patch_size=32,
    num_rings=5,
    patches_per_ring=8,
    min_crop=32,
    max_crop=256,
):
    """
    Sample patches from an image with foveated density.

    Near the center: small crops (high detail) packed densely.
    Far from center: large crops (low detail) spaced apart.
    Every patch is resized to the same (patch_size x patch_size) —
    this is one token for the transformer.

    Returns
    -------
    patches : list of (patch_size, patch_size, 3) arrays — the tokens
    meta    : list of dicts with {x, y, crop_size, ring} for each patch
    """
    h, w = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    cx, cy = center

    max_radius = min(cx, cy, w - cx, h - cy) * 0.95

    # Ring radii: 0 (center patch) + exponentially spaced rings
    ring_radii = [0] + list(
        np.logspace(np.log10(max_radius * 0.08), np.log10(max_radius), num_rings)
    )

    # Crop size grows with ring (inner = small crop = high detail)
    crop_sizes = np.linspace(min_crop, max_crop, len(ring_radii)).astype(int)

    patches = []
    meta = []

    for ring_idx, (radius, crop_sz) in enumerate(zip(ring_radii, crop_sizes)):
        if ring_idx == 0:
            # Center patch
            angles = [0]
        else:
            angles = np.linspace(0, 2 * np.pi, patches_per_ring, endpoint=False)
            # Offset every other ring so patches interleave
            if ring_idx % 2 == 1:
                angles += np.pi / patches_per_ring

        for angle in angles:
            px = int(cx + radius * np.cos(angle))
            py = int(cy + radius * np.sin(angle))

            # Crop bounds
            half = crop_sz // 2
            x1 = max(0, px - half)
            y1 = max(0, py - half)
            x2 = min(w, px + half)
            y2 = min(h, py + half)

            if x2 - x1 < 4 or y2 - y1 < 4:
                continue

            crop = image[y1:y2, x1:x2]
            # Resize to uniform token size
            token = cv2.resize(
                crop, (patch_size, patch_size), interpolation=cv2.INTER_AREA
            )

            patches.append(token)
            meta.append(
                {
                    "x": px,
                    "y": py,
                    "crop_size": crop_sz,
                    "ring": ring_idx,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )

    return patches, meta


def visualize(image, patches, meta, patch_size=32):
    """Draw where each patch comes from on the image, and show the token grid."""
    h, w = image.shape[:2]

    # --- Left: source image with patch regions drawn ---
    vis_src = image.copy()
    colors = [
        (0, 255, 0),  # ring 0 - green (fovea)
        (0, 200, 255),  # ring 1 - yellow
        (0, 150, 255),  # ring 2 - orange
        (0, 80, 255),  # ring 3 - red-orange
        (0, 0, 255),  # ring 4 - red
        (255, 0, 255),  # ring 5 - magenta
    ]
    for m in meta:
        c = colors[min(m["ring"], len(colors) - 1)]
        cv2.rectangle(vis_src, (m["x1"], m["y1"]), (m["x2"], m["y2"]), c, 2)
        cv2.circle(vis_src, (m["x"], m["y"]), 3, c, -1)

    # --- Right: the actual tokens (what the model sees) ---
    n = len(patches)
    cols = min(n, 8)
    rows = (n + cols - 1) // cols
    gap = 2
    grid_w = cols * (patch_size + gap) + gap
    grid_h = rows * (patch_size + gap) + gap
    vis_tokens = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 40

    for i, (patch, m) in enumerate(zip(patches, meta)):
        r, c_idx = divmod(i, cols)
        y = gap + r * (patch_size + gap)
        x = gap + c_idx * (patch_size + gap)
        vis_tokens[y : y + patch_size, x : x + patch_size] = patch
        # Color border to match ring
        clr = colors[min(m["ring"], len(colors) - 1)]
        cv2.rectangle(
            vis_tokens, (x - 1, y - 1), (x + patch_size, y + patch_size), clr, 1
        )

    return vis_src, vis_tokens


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--cx", type=int, default=None)
    parser.add_argument("--cy", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=48)
    parser.add_argument("--rings", type=int, default=5)
    parser.add_argument("--per-ring", type=int, default=10)
    parser.add_argument("--min-crop", type=int, default=32)
    parser.add_argument("--max-crop", type=int, default=200)
    parser.add_argument("-o", "--output", default="patches")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    h, w = img.shape[:2]
    center = (args.cx or w // 2, args.cy or h // 2)

    patches, meta = sample_foveated_patches(
        img,
        center,
        args.patch_size,
        args.rings,
        args.per_ring,
        args.min_crop,
        args.max_crop,
    )

    vis_src, vis_tokens = visualize(img, patches, meta, args.patch_size)

    cv2.imwrite(f"{args.output}_source.png", vis_src)
    cv2.imwrite(f"{args.output}_tokens.png", vis_tokens)

    print(f"Sampled {len(patches)} patches of {args.patch_size}x{args.patch_size}")
    print(f"Total token pixels: {len(patches) * args.patch_size**2:,}")
    print(f"Original pixels:    {h * w:,}")
    print(f"Compression:        {h * w / (len(patches) * args.patch_size**2):.1f}x")
    for ring in range(args.rings + 1):
        ring_patches = [m for m in meta if m["ring"] == ring]
        if ring_patches:
            cs = ring_patches[0]["crop_size"]
            print(
                f"  Ring {ring}: {len(ring_patches)} patches, "
                f"crop {cs}x{cs} → {args.patch_size}x{args.patch_size} "
                f"(effective {cs / args.patch_size:.1f}x downsample)"
            )


if __name__ == "__main__":
    main()
