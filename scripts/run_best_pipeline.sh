#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPU_LIST="${GPU_LIST:-0,1,2,3}"
NP="${NP:-4}"
INIT_CKPT="experiments/_bak_official_75.01/cvm_checkpoint_best.pkl"

if [[ ! -f "$INIT_CKPT" ]]; then
  echo "Missing initialization checkpoint: $INIT_CKPT" >&2
  exit 2
fi

echo "[best] train CVM-002 on GPUs $GPU_LIST (NP=$NP)"
CUDA_VISIBLE_DEVICES="$GPU_LIST" NP="$NP" \
  bash scripts/train_ddp.sh configs/task/train_spcfgfncvm002_cvm.yaml

echo "[best] train SPCF distance head"
CUDA_VISIBLE_DEVICES="$GPU_LIST" NP="$NP" \
  bash scripts/train_ddp.sh configs/task/train_spcfgfncvm002.yaml

echo "[best] complete"
