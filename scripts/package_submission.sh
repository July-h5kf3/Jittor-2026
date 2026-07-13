#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPU="${GPU:-0}"
OUT_DIR="submission_results/cvm002_spcfgfncvm002"
ZIP_PATH="submission_results/result_cvm002_spcfgfncvm002_rebuilt.zip"

if [[ -e "$OUT_DIR" ]]; then
  echo "$OUT_DIR already exists; remove it explicitly before rebuilding." >&2
  exit 2
fi

CUDA_VISIBLE_DEVICES="$GPU" \
  python run.py --task configs/task/predict_spcfgfncvm002.yaml

COUNT="$(find "$OUT_DIR" -type f -name denoised.npy | wc -l | tr -d ' ')"
if [[ "$COUNT" != "200" ]]; then
  echo "Expected 200 denoised.npy files, found $COUNT" >&2
  exit 3
fi

(cd "$OUT_DIR" && zip -qr "$ROOT/$ZIP_PATH" shapenet)
echo "Created $ZIP_PATH with $COUNT predictions"
