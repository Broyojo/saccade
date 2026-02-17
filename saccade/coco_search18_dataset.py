"""
COCO-Search18 (TP + TA) trial dataset.

Each item is a full scanpath (one trial):
  patches           FloatTensor [T, 3, pH, pW]   one patch per fixation
  saccade_amplitude FloatTensor [T-1]           per-movement amplitude (fix i -> i+1)
  target            str                         1 of 18 categories
  found             bool                        whether target was found (TP only)
  condition         str                         "present" or "absent"

Notes
- Fixation coordinates are in the original 1680x1050 pixel space.
- If available, TP saccade amplitudes are read from the supplemental
  `extras/saccade_amplitude_TP_trainval.npy` (Google Drive file is mislabeled
  as .zip in some download scripts). For trials without supplemental amplitude,
  amplitudes are approximated from pixel distances using a calibrated px/deg
  factor.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import warnings
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import normalize, to_tensor

ORIG_W, ORIG_H = 1680, 1050

CATEGORIES = [
    "bottle",
    "bowl",
    "car",
    "chair",
    "clock",
    "cup",
    "fork",
    "keyboard",
    "knife",
    "laptop",
    "microwave",
    "mouse",
    "oven",
    "potted plant",
    "sink",
    "stop sign",
    "toilet",
    "tv",
]


def _as_list(x):
    if isinstance(x, list):
        return x
    if hasattr(x, "tolist"):
        return x.tolist()
    return list(x)


def _image_root(tp_or_ta_dir: Path) -> Path:
    # COCO-Search18 zips usually extract to: images/{TP,TA}/images/<category>/*.jpg
    return tp_or_ta_dir / "images" if (tp_or_ta_dir / "images").is_dir() else tp_or_ta_dir


def _first_existing(*paths: Path) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


class COCOSearch18Dataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",  # "train" | "valid" | "all"
        condition: str = "all",  # "TP" | "TA" | "all"
        patch_size: tuple[int, int] = (224, 224),  # (H, W)
        amplitude_default_px_per_deg: float = 32.0,
    ):
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size

        self.split = self._norm_split(split)
        self.condition = condition.strip().upper()
        if self.condition not in {"TP", "TA", "ALL"}:
            raise ValueError("condition must be 'TP', 'TA', or 'all'")

        self.tp_image_dir = _image_root(self.data_dir / "images" / "TP")
        self.ta_image_dir = _image_root(self.data_dir / "images" / "TA")

        self._tp_image_index = None
        self._ta_image_index = None

        if self.condition in {"TP", "ALL"} and not any(self.tp_image_dir.rglob("*.jpg")):
            raise FileNotFoundError(
                f"TP images not found under: {self.tp_image_dir}\n"
                "Run: bash download_coco_search18.sh"
            )
        if self.condition in {"TA", "ALL"} and not any(self.ta_image_dir.rglob("*.jpg")):
            raise FileNotFoundError(
                f"TA images not found under: {self.ta_image_dir}\n"
                "Run: bash download_coco_search18.sh"
            )

        self.amp_lookup, self.px_per_deg = self._load_tp_amplitude_lookup(
            default_px_per_deg=amplitude_default_px_per_deg
        )

        self.trials = []
        if self.condition in {"TP", "ALL"}:
            self.trials.extend(self._load_tp_fixations())
        if self.condition in {"TA", "ALL"}:
            self.trials.extend(self._load_ta_fixations())

        # Keep only requested split.
        if self.split != "all":
            self.trials = [sp for sp in self.trials if sp.get("split") == self.split]

        # Preprocess + filter unusable trials.
        kept = []
        for sp in self.trials:
            xs = _as_list(sp["X"])
            ys = _as_list(sp["Y"])
            if len(xs) < 2 or len(ys) < 2:
                continue
            # clip to bounds
            xs = [max(0.0, min(float(x), ORIG_W - 1.0)) for x in xs]
            ys = [max(0.0, min(float(y), ORIG_H - 1.0)) for y in ys]
            kept.append({**sp, "X": xs, "Y": ys, "length": len(xs)})
        self.trials = kept

    @staticmethod
    def _norm_split(split: str) -> str:
        s = str(split).strip().lower()
        if s in {"val", "validation"}:
            return "valid"
        if s in {"train", "valid", "all"}:
            return s
        raise ValueError("split must be 'train', 'valid' (or val/validation), or 'all'")

    def _load_tp_fixations(self) -> list[dict]:
        fix_dir = self.data_dir / "fixations" / "TP"
        if not fix_dir.is_dir():
            raise FileNotFoundError(
                f"Missing TP fixations dir: {fix_dir}\nRun: bash download_coco_search18.sh"
            )
        paths = sorted(fix_dir.glob("coco_search18_fixations_TP_*_split*.json"))
        if not paths:
            raise FileNotFoundError(
                f"No TP fixation jsons found under: {fix_dir}\nRun: bash download_coco_search18.sh"
            )
        # The official release includes split1/split2 files with heavy overlap.
        # Deduplicate by (name, subject, task, split) to get the union.
        out = {}
        for p in paths:
            with open(p) as f:
                for sp in json.load(f):
                    key = (sp["name"], int(sp["subject"]), sp["task"], sp["split"])
                    out.setdefault(key, sp)
        return list(out.values())

    def _load_ta_fixations(self) -> list[dict]:
        p = _first_existing(
            self.data_dir / "fixations" / "TA" / "coco_search18_fixations_TA_trainval.json",
            self.data_dir / "fixations" / "coco_search18_fixations_TA_trainval.json",
        )
        if p is None:
            raise FileNotFoundError(
                "Missing TA fixations json.\nRun: bash download_coco_search18.sh"
            )
        with open(p) as f:
            return json.load(f)

    def _build_image_index(self, root: Path) -> dict[str, Path]:
        if not root.is_dir():
            return {}
        return {p.name: p for p in root.rglob("*.jpg")}

    def _get_image_path(self, sp: dict) -> Path:
        name = sp["name"]
        if sp.get("condition") == "present":
            if self._tp_image_index is None:
                self._tp_image_index = self._build_image_index(self.tp_image_dir)
            p = self._tp_image_index.get(name)
            if p is None:
                raise FileNotFoundError(
                    f"Missing TP image: {name}\nExpected under: {self.tp_image_dir}\n"
                    "Run: bash download_coco_search18.sh"
                )
            return p

        if self._ta_image_index is None:
            self._ta_image_index = self._build_image_index(self.ta_image_dir)
        p = self._ta_image_index.get(name)
        if p is None:
            raise FileNotFoundError(
                f"Missing TA image: {name}\nExpected under: {self.ta_image_dir}\n"
                "Run: bash download_coco_search18.sh"
            )
        return p

    def _load_tp_amplitude_lookup(self, default_px_per_deg: float) -> tuple[dict, float]:
        amp_path = _first_existing(
            self.data_dir / "extras" / "saccade_amplitude_TP_trainval.npy",
            self.data_dir / "extras" / "saccade_amplitude_TP_trainval.zip",  # actually .npy
        )
        if amp_path is None:
            return {}, float(default_px_per_deg)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=getattr(np, "VisibleDeprecationWarning", Warning)
            )
            arr = np.load(amp_path, allow_pickle=True)
        lookup: dict[tuple[str, int, str, str], np.ndarray] = {}
        ratios = []

        for d in arr:
            key = (d["name"], int(d["subject"]), d["task"], d["split"])
            amp = np.asarray(d["amplitude"], dtype=np.float32)

            xs0 = np.asarray(d["X_start"], dtype=np.float32)
            ys0 = np.asarray(d["Y_start"], dtype=np.float32)
            xs1 = np.asarray(d["X_end"], dtype=np.float32)
            ys1 = np.asarray(d["Y_end"], dtype=np.float32)

            n = min(len(amp), len(xs0), len(ys0), len(xs1), len(ys1))
            if n <= 0:
                continue
            amp = amp[:n]
            xs0, ys0, xs1, ys1 = xs0[:n], ys0[:n], xs1[:n], ys1[:n]
            lookup[key] = amp

            dist = np.hypot(xs1 - xs0, ys1 - ys0)
            m = (amp > 1e-6) & np.isfinite(dist) & (dist > 0)
            if m.any():
                ratios.append(dist[m] / amp[m])

        if not ratios:
            return lookup, float(default_px_per_deg)

        px_per_deg = float(np.median(np.concatenate(ratios)))
        if not np.isfinite(px_per_deg) or px_per_deg <= 0:
            px_per_deg = float(default_px_per_deg)
        return lookup, px_per_deg

    def _crop_patch(self, img_tensor: torch.Tensor, cx: float, cy: float) -> torch.Tensor:
        pH, pW = self.patch_size
        pad_h, pad_w = pH // 2 + 1, pW // 2 + 1
        img_padded = F.pad(img_tensor, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
        cx_p = int(round(cx)) + pad_w
        cy_p = int(round(cy)) + pad_h
        top = cy_p - (pH // 2)
        left = cx_p - (pW // 2)
        return img_padded[:, top : top + pH, left : left + pW]

    def __len__(self) -> int:
        return len(self.trials)

    def __getitem__(self, idx: int) -> dict:
        sp = self.trials[idx]

        img = to_tensor(Image.open(self._get_image_path(sp)).convert("RGB"))
        img = normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        xs = sp["X"]
        ys = sp["Y"]
        patches = torch.stack([self._crop_patch(img, xs[i], ys[i]) for i in range(len(xs))])
        fixation_xy = torch.tensor(
            [[int(round(xs[i])), int(round(ys[i]))] for i in range(len(xs))],
            dtype=torch.int64,
        )

        key = (sp["name"], int(sp["subject"]), sp["task"], sp["split"])
        amps = None
        if sp.get("condition") == "present":
            amps = self.amp_lookup.get(key)
        if amps is not None and len(amps) == len(xs) - 1:
            sacc_amp = torch.tensor(amps, dtype=torch.float32)
        else:
            x = np.asarray(xs, dtype=np.float32)
            y = np.asarray(ys, dtype=np.float32)
            dist_px = np.hypot(x[1:] - x[:-1], y[1:] - y[:-1])
            sacc_amp = torch.from_numpy(dist_px / float(self.px_per_deg)).to(torch.float32)

        present = sp.get("condition") == "present"
        found = bool(present and sp.get("correct", 0) == 1)

        return {
            "patches": patches,  # [T, 3, pH, pW]
            "fixation_xy": fixation_xy,  # [T, 2] int (x, y) in original 1680x1050 space
            "saccade_amplitude": sacc_amp,  # [T-1]
            "target": sp["task"],
            "found": found,
            "condition": sp.get("condition", "present" if present else "absent"),
        }


def collate_fn(batch: list[dict]) -> dict:
    max_t = max(b["patches"].shape[0] for b in batch)
    C, pH, pW = batch[0]["patches"].shape[1:]

    patches = torch.zeros(len(batch), max_t, C, pH, pW)
    patch_mask = torch.zeros(len(batch), max_t, dtype=torch.bool)
    fixation_xy = torch.zeros(len(batch), max_t, 2, dtype=torch.int64)

    # amplitude has length (T-1)
    max_s = max_t - 1
    sacc_amp = torch.zeros(len(batch), max_s, dtype=torch.float32)
    sacc_mask = torch.zeros(len(batch), max_s, dtype=torch.bool)

    for i, b in enumerate(batch):
        t = b["patches"].shape[0]
        patches[i, :t] = b["patches"]
        patch_mask[i, :t] = True
        fixation_xy[i, :t] = b["fixation_xy"]

        s = b["saccade_amplitude"].shape[0]
        sacc_amp[i, :s] = b["saccade_amplitude"]
        sacc_mask[i, :s] = True

    return {
        "patches": patches,
        "patch_mask": patch_mask,
        "fixation_xy": fixation_xy,
        "saccade_amplitude": sacc_amp,
        "saccade_mask": sacc_mask,
        "target": [b["target"] for b in batch],
        "found": torch.tensor([b["found"] for b in batch], dtype=torch.bool),
        "condition": [b["condition"] for b in batch],
    }


if __name__ == "__main__":
    data_dir = Path("./data/coco_search18")

    # Keep this runnable even if TA images aren't downloaded yet.
    tp_ok = any(_image_root(data_dir / "images" / "TP").rglob("*.jpg"))
    ta_ok = any(_image_root(data_dir / "images" / "TA").rglob("*.jpg"))
    cond = "all" if (tp_ok and ta_ok) else ("TP" if tp_ok else "TA")

    ds = COCOSearch18Dataset(data_dir, split="train", condition=cond)
    print(f"Trials: {len(ds)}  (split=train, condition={cond}, px/deg~{ds.px_per_deg:.2f})")

    loader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(loader))
    for k, v in batch.items():
        print(f"  {k}: {tuple(v.shape) if hasattr(v, 'shape') else v}")
