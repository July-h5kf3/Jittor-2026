#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
SUBMIT_DEPS="${SUBMIT_DEPS:-/root/data-tmp/submit_deps}"

PNX_CKPT="experiments/spcfgfnpnx001_spcf/checkpoint_best.pkl"
PNX_CKPT_SHA256="726176761d49b61eb4b8de8aad66bb7f01f51ee57ee3b88f71657fc7831b7708"
ROT_FUSED_DIR="submission_results/rot001_a105_cv_twopass_g050_20260719"
RAW_DIR="submission_results/pnx001_a105_raw_20260720"
ADAPTIVE_DIR="submission_results/pnx001_a105_adaptive_20260720"
PASS2_DIR="submission_results/pnx001_a105_pass2_20260720"
FUSED_DIR="submission_results/pnx001_a105_cv_twopass_g050_20260720"
ENSEMBLE_DIR="submission_results/rot001_pnx001_twopass_ens_p030_20260720"
ZIP_PATH="submission_results/result_rot001_pnx001_twopass_ens_p030_20260720.zip"
ALPHA_MANIFEST="submission_results/pnx001_a105_adaptive_20260720.tsv"
FUSION_MANIFEST="submission_results/pnx001_a105_cv_twopass_g050_20260720.tsv"

[[ -f "$PNX_CKPT" ]] || { echo "Missing PNX checkpoint: $PNX_CKPT" >&2; exit 2; }
actual_sha="$(sha256sum "$PNX_CKPT" | awk '{print $1}')"
if [[ "$actual_sha" != "$PNX_CKPT_SHA256" ]]; then
  echo "PNX checkpoint SHA256 mismatch: $actual_sha" >&2
  exit 2
fi

rot_count="$(find "$ROT_FUSED_DIR" -type f -name denoised.npy | wc -l)"
if [[ "$rot_count" -ne 200 ]]; then
  echo "Expected 200 ROT fused predictions, found $rot_count" >&2
  exit 2
fi

for target in \
  "$RAW_DIR" \
  "$ADAPTIVE_DIR" \
  "$PASS2_DIR" \
  "$FUSED_DIR" \
  "$ENSEMBLE_DIR" \
  "$ZIP_PATH" \
  "$ZIP_PATH.sha256" \
  "$ALPHA_MANIFEST" \
  "$FUSION_MANIFEST"; do
  if [[ -e "$target" ]]; then
    echo "Refusing to overwrite existing PNX candidate artifact: $target" >&2
    exit 3
  fi
done

export TMPDIR="${TMPDIR:-/root/data-tmp/tmp}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export CUDA_VISIBLE_DEVICES="$GPU"
export use_mpi=0
mkdir -p "$TMPDIR" submission_results

echo "[PNX candidate] canonical raw prediction"
"$PYTHON_BIN" run.py \
  --task configs/task/predict_spcfgfnpnx001a105_submit.yaml \
  --seed 123

echo "[PNX candidate] label-free mean/var calibration"
"$PYTHON_BIN" scripts/calibrate_predictions.py \
  --pred-dir "$RAW_DIR" \
  --noisy-dir . \
  --out-dir "$ADAPTIVE_DIR" \
  --profile mean-var \
  --expected-count 200 \
  --manifest "$ALPHA_MANIFEST"

echo "[PNX candidate] canonical second pass"
"$PYTHON_BIN" run.py \
  --task configs/task/predict_spcfgfnpnx001a105_submit_pass2.yaml \
  --seed 123

echo "[PNX candidate] CV-adaptive two-pass fusion"
"$PYTHON_BIN" scripts/calibrate_adaptive_two_pass.py \
  --first-dir "$RAW_DIR" \
  --adaptive-dir "$ADAPTIVE_DIR" \
  --second-dir "$PASS2_DIR" \
  --out-dir "$FUSED_DIR" \
  --alpha-manifest "$ALPHA_MANIFEST" \
  --manifest "$FUSION_MANIFEST" \
  --expected-count 200 \
  --score-feature cv \
  --gamma 0.50

echo "[PNX candidate] fixed 70/30 ROT-PNX output ensemble"
"$PYTHON_BIN" scripts/ensemble_predictions.py \
  --input-dir "$ROT_FUSED_DIR" \
  --input-dir "$FUSED_DIR" \
  --weights 0.70,0.30 \
  --output-dir "$ENSEMBLE_DIR" \
  --expected-count 200

echo "[PNX candidate] strict array validation"
"$PYTHON_BIN" - "$ENSEMBLE_DIR" <<'PY'
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
files = sorted(root.rglob("denoised.npy"))
noisy_root = Path("dataset_test_noisy")
errors = []
for path in files:
    relative = path.parent.relative_to(root)
    parts = relative.parts
    if "shapenet" not in parts:
        errors.append(f"noncanonical:{relative}")
        continue
    index = parts.index("shapenet")
    noisy = noisy_root.joinpath(*parts[index:], "noisy.npy")
    points = np.load(path, allow_pickle=False)
    source = np.load(noisy, allow_pickle=False)
    if points.dtype != np.float32:
        errors.append(f"dtype:{relative}:{points.dtype}")
    if points.shape != source.shape or points.shape != (50000, 3):
        errors.append(f"shape:{relative}:{points.shape}:{source.shape}")
    if not np.isfinite(points).all():
        errors.append(f"nonfinite:{relative}")
if len(files) != 200 or errors:
    raise SystemExit(f"validation failed: count={len(files)} errors={errors[:5]}")
print("validated 200 float32 finite arrays with input-matched shape")
PY

echo "[PNX candidate] package archive"
SKIP_PREDICT=1 \
TASK_CONFIG=configs/task/predict_spcfgfnpnx001a105_submit.yaml \
OUT_DIR="$ENSEMBLE_DIR" \
ZIP_PATH="$ZIP_PATH" \
EXPECTED_COUNT=200 \
EXPECTED_POINTS=50000 \
bash scripts/package_submission.sh

echo "[PNX candidate] strict archive validation"
PYTHONPATH="$SUBMIT_DEPS${PYTHONPATH:+:$PYTHONPATH}" \
"$PYTHON_BIN" -c \
  'import sys; from pathlib import Path; from scripts.submit_educoder import validate_archive; print(validate_archive(Path(sys.argv[1]), 200, 50000, 10.0))' \
  "$ZIP_PATH"

echo "[PNX candidate] ready: $ZIP_PATH"
