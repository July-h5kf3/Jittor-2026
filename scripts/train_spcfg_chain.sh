#!/usr/bin/env bash
# Sequential 3-stage StraightPCF training: VM -> CVM -> SPCF, then local eval.
# Each stage runs 2-GPU (mpirun -np 2) to completion (early stopping), chaining
# checkpoints via init_stage_ckpt in the task configs. Run with nohup.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=/root/miniconda3/bin:$PATH

TS=$(date +%Y%m%d_%H%M%S)
MASTER=train_logs/spcfg_chain_${TS}.log
mkdir -p train_logs
echo "[chain] start $(date)" | tee -a "$MASTER"

# fresh start
rm -rf experiments/spcfg_vm experiments/spcfg_cvm experiments/spcfg_spcf

run_stage () {
  local task=$1 name=$2 ckptdir=$3
  echo "[chain] === STAGE ${name} start $(date) ===" | tee -a "$MASTER"
  pkill -9 -f 'run.py --task' 2>/dev/null; pkill -9 -f mpirun 2>/dev/null; sleep 3
  NP=2 bash scripts/train_ddp.sh "configs/task/${task}" > "train_logs/stageg_${name}_${TS}.log" 2>&1
  pkill -9 -f 'run.py --task' 2>/dev/null; pkill -9 -f mpirun 2>/dev/null; sleep 3
  if [ ! -f "${ckptdir}/checkpoint_best.pkl" ]; then
    echo "[chain] FAIL: no checkpoint for stage ${name}" | tee -a "$MASTER"; exit 1
  fi
  echo "[chain] ${name} done $(date), ckpt size $(stat -c%s ${ckptdir}/checkpoint_best.pkl)" | tee -a "$MASTER"
}

run_stage train_spcfg_vm.yaml   vm   experiments/spcfg_vm
run_stage train_spcfg_cvm.yaml  cvm  experiments/spcfg_cvm
run_stage train_spcfg.yaml      spcf experiments/spcfg_spcf

echo "[chain] === PREDICT + EVAL on localtest === $(date)" | tee -a "$MASTER"
pkill -9 -f 'run.py --task' 2>/dev/null; sleep 2
rm -rf results_local
python run.py --task configs/task/predict_spcfg_local.yaml > "train_logs/predict_spcf_local_${TS}.log" 2>&1
python evaluate.py --pred_dir results_local/localtest --gt_dir localtest --noisy_dir localtest \
    --mesh_dir localtest --workers 16 2>&1 | tee "train_logs/spcf_local_score_${TS}.txt" | tee -a "$MASTER"
echo "[chain] ALL DONE $(date)" | tee -a "$MASTER"
