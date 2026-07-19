#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
SUBMIT_DEPS="${SUBMIT_DEPS:-/root/data-tmp/submit_deps}"

RAW_DIR="submission_results/rot001_a105_raw_20260719"
ADAPTIVE_DIR="submission_results/rot001_a105_adaptive_20260719"
PASS2_DIR="submission_results/rot001_a105_pass2_20260719"
FUSED_DIR="submission_results/rot001_a105_cv_twopass_g050_20260719"
ZIP_PATH="submission_results/result_rot001_a105_cv_twopass_g050_20260719.zip"
ALPHA_MANIFEST="submission_results/rot001_a105_adaptive_20260719.tsv"
FUSION_MANIFEST="submission_results/rot001_a105_cv_twopass_g050_20260719.tsv"

for target in \
  "$RAW_DIR" \
  "$ADAPTIVE_DIR" \
  "$PASS2_DIR" \
  "$FUSED_DIR" \
  "$ZIP_PATH" \
  "$ZIP_PATH.sha256"; do
  if [[ -e "$target" ]]; then
    echo "Refusing to overwrite existing ROT submission artifact: $target" >&2
    exit 2
  fi
done

export HOME="${HOME_OVERRIDE:-/root/data-tmp/jittor_home_rot001_submit}"
export TMPDIR="${TMPDIR:-/root/data-tmp/tmp}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export CUDA_VISIBLE_DEVICES="$GPU"
export use_mpi=0
mkdir -p "$HOME" "$TMPDIR" submission_results

echo "[ROT submit] canonical raw prediction"
"$PYTHON_BIN" run.py \
  --task configs/task/predict_spcfgfnrot001a105_submit.yaml \
  --seed 123

echo "[ROT submit] label-free mean/var calibration"
"$PYTHON_BIN" scripts/calibrate_predictions.py \
  --pred-dir "$RAW_DIR" \
  --noisy-dir . \
  --out-dir "$ADAPTIVE_DIR" \
  --profile mean-var \
  --expected-count 200 \
  --manifest "$ALPHA_MANIFEST"

echo "[ROT submit] canonical second pass"
"$PYTHON_BIN" run.py \
  --task configs/task/predict_spcfgfnrot001a105_submit_pass2.yaml \
  --seed 123

echo "[ROT submit] CV-adaptive two-pass fusion"
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

echo "[ROT submit] package archive"
SKIP_PREDICT=1 \
TASK_CONFIG=configs/task/predict_spcfgfnrot001a105_submit.yaml \
OUT_DIR="$FUSED_DIR" \
ZIP_PATH="$ZIP_PATH" \
EXPECTED_COUNT=200 \
EXPECTED_POINTS=50000 \
bash scripts/package_submission.sh

echo "[ROT submit] strict archive validation"
PYTHONPATH="$SUBMIT_DEPS${PYTHONPATH:+:$PYTHONPATH}" \
"$PYTHON_BIN" -c \
  'import sys; from pathlib import Path; from scripts.submit_educoder import validate_archive; print(validate_archive(Path(sys.argv[1]), 200, 50000, 10.0))' \
  "$ZIP_PATH"

echo "[ROT submit] ready: $ZIP_PATH"
