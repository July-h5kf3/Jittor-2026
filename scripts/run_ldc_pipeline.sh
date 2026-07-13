#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPU_LIST="${GPU_LIST:-0,1,2,3}"
NP="${NP:-4}"
PRED_GPU="${PRED_GPU:-4}"
BASE_CKPT="experiments/spcfgfncvm002_spcf/checkpoint_best.pkl"
LDC_DIR="experiments/spcfgfnldc_spcf"
RESULT_DIR="results_local2_fnldc"
SCORE_FILE="train_logs/spcfgfnldc_local2_score.txt"

[[ -f "$BASE_CKPT" ]] || { echo "Missing $BASE_CKPT" >&2; exit 2; }
[[ ! -e "$RESULT_DIR" ]] || { echo "$RESULT_DIR already exists" >&2; exit 3; }

mkdir -p "$LDC_DIR" train_logs
# Keep a guaranteed baseline fallback. LDC changes the validation objective by
# matching the two inference iterations, so its loss is not directly comparable
# with the historical single-unroll validation metric.
cp -f -- "$BASE_CKPT" "$LDC_DIR/checkpoint_baseline.pkl"
rm -f -- "$LDC_DIR/checkpoint_best.pkl"

export HOME="${HOME_OVERRIDE:-/root/data-tmp/jittor_home_cvm002}"
export cache_path="${cache_path:-/root/data-tmp/.cache/jittor_ldc}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

echo "[LDC] fine-tune matched-unroll distance conditioning on GPUs $GPU_LIST"
CUDA_VISIBLE_DEVICES="$GPU_LIST" NP="$NP" \
  bash scripts/train_ddp.sh configs/task/train_spcfgfnldc.yaml

if [[ ! -f "$LDC_DIR/checkpoint_best.pkl" ]]; then
  echo "[LDC] no new checkpoint; use baseline fallback"
  cp -f -- "$LDC_DIR/checkpoint_baseline.pkl" "$LDC_DIR/checkpoint_best.pkl"
fi

echo "[LDC] predict localtest2 on GPU $PRED_GPU"
CUDA_VISIBLE_DEVICES="$PRED_GPU" use_mpi=0 \
  /root/miniconda3/bin/python run.py --task configs/task/predict_spcfgfnldc_local2.yaml

COUNT="$(find "$RESULT_DIR" -type f -name denoised.npy | wc -l | tr -d ' ')"
[[ "$COUNT" == "62" ]] || { echo "Expected 62 predictions, found $COUNT" >&2; exit 4; }

/root/miniconda3/bin/python evaluate.py \
  --pred_dir "$RESULT_DIR/localtest2" \
  --gt_dir localtest2 \
  --noisy_dir localtest2 \
  --mesh_dir localtest2 \
  --workers 16 | tee "$SCORE_FILE"

echo "[LDC] completed; score saved to $SCORE_FILE"
