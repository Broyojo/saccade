#!/usr/bin/env bash
# Download + extract COCO-Search18 (TP + TA) and the TP saccade-amplitude file.
#
# Usage:
#   bash download_coco_search18.sh [--data-dir ./data/coco_search18]
#
# Notes:
# - The "saccade_amplitude_TP_trainval" Google Drive file is a NumPy .npy object
#   array (not a zip). We save it as .npy.

set -euo pipefail

DATA_DIR="./data/coco_search18"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--data-dir ./data/coco_search18]"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "Error: missing '$1' in PATH" >&2; exit 1; }
}

download() {
  local url="$1" dest="$2"
  [[ -f "$dest" ]] && return
  mkdir -p "$(dirname "$dest")"
  echo "[download] $dest"
  wget --progress=bar:force:noscroll -O "$dest" "$url"
}

extract_zip() {
  local zip="$1" dest_dir="$2" sentinel="$3"
  [[ -e "$sentinel" ]] && return
  mkdir -p "$dest_dir"
  echo "[extract]  $zip -> $dest_dir"
  unzip -q "$zip" -d "$dest_dir"
}

need_cmd wget
need_cmd unzip

SBU_BASE="http://vision.cs.stonybrook.edu/~cvlab_download"
AMP_ID="1Aa3hZUZ0Jf1YnR_p7-zhcHRwBW0nODwP"

mkdir -p "$DATA_DIR/images/TP" "$DATA_DIR/images/TA" "$DATA_DIR/fixations/TP" "$DATA_DIR/fixations/TA" "$DATA_DIR/extras"

download "$SBU_BASE/COCOSearch18-images-TP.zip" "$DATA_DIR/COCOSearch18-images-TP.zip"
download "$SBU_BASE/COCOSearch18-images-TA.zip" "$DATA_DIR/COCOSearch18-images-TA.zip"
download "$SBU_BASE/COCOSearch18-fixations-TP.zip" "$DATA_DIR/fixations/COCOSearch18-fixations-TP.zip"
TA_FIX_OUT="$DATA_DIR/fixations/TA/coco_search18_fixations_TA_trainval.json"
if [[ ! -f "$TA_FIX_OUT" ]] && [[ -f "$DATA_DIR/fixations/coco_search18_fixations_TA_trainval.json" ]]; then
  echo "[move]     $DATA_DIR/fixations/coco_search18_fixations_TA_trainval.json -> $TA_FIX_OUT"
  mv "$DATA_DIR/fixations/coco_search18_fixations_TA_trainval.json" "$TA_FIX_OUT"
fi
download "$SBU_BASE/coco_search18_fixations_TA_trainval.json" "$TA_FIX_OUT"

extract_zip "$DATA_DIR/COCOSearch18-images-TP.zip" "$DATA_DIR/images/TP" "$DATA_DIR/images/TP/images"
extract_zip "$DATA_DIR/COCOSearch18-images-TA.zip" "$DATA_DIR/images/TA" "$DATA_DIR/images/TA/images"
extract_zip "$DATA_DIR/fixations/COCOSearch18-fixations-TP.zip" "$DATA_DIR/fixations/TP" "$DATA_DIR/fixations/TP/coco_search18_fixations_TP_train_split1.json"

AMP_OUT="$DATA_DIR/extras/saccade_amplitude_TP_trainval.npy"
if [[ ! -f "$AMP_OUT" ]] && [[ -f "$DATA_DIR/extras/saccade_amplitude_TP_trainval.zip" ]]; then
  echo "[move]     $DATA_DIR/extras/saccade_amplitude_TP_trainval.zip -> $AMP_OUT"
  mv "$DATA_DIR/extras/saccade_amplitude_TP_trainval.zip" "$AMP_OUT"
fi
if [[ ! -f "$AMP_OUT" ]]; then
  if command -v uv >/dev/null 2>&1; then
    uv run gdown "https://drive.google.com/uc?id=$AMP_ID" -O "$AMP_OUT"
  elif command -v gdown >/dev/null 2>&1; then
    gdown "https://drive.google.com/uc?id=$AMP_ID" -O "$AMP_OUT"
  else
    echo "Error: missing 'gdown' to fetch the amplitude file." >&2
    echo "Install with: uv sync  (or: uv pip install gdown)" >&2
    exit 1
  fi
fi

echo "Done: $DATA_DIR"
