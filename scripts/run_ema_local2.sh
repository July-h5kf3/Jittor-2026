#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPU_LIST="${GPU_LIST:-2,3,4,5}"
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if [[ "${#GPUS[@]}" -ne 4 ]]; then
  echo "GPU_LIST must contain exactly four comma-separated GPU IDs" >&2
  exit 2
fi

export HOME="${HOME_OVERRIDE:-/root/data-tmp/jittor_home_cvm002}"
export cache_path="${cache_path:-/root/data-tmp/jittor_cache}"
export TMPDIR="${TMPDIR:-/root/data-tmp/jittor_tmp_ema_predict}"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export DISABLE_MULTIPROCESSING="${DISABLE_MULTIPROCESSING:-1}"
export use_parallel_op_compiler="${use_parallel_op_compiler:-0}"
export OMPI_MCA_orte_tmpdir_base="${OMPI_MCA_orte_tmpdir_base:-/dev/shm/ema-predict}"
export OMPI_MCA_btl_vader_backing_directory="${OMPI_MCA_btl_vader_backing_directory:-/dev/shm}"
mkdir -p "$TMPDIR" "$OMPI_MCA_orte_tmpdir_base" /root/log/2026-07-14

labels=(raw ema099 ema0995 ema0999)
tasks=(
  configs/task/predict_spcfgfncvm002ema_raw_local2.yaml
  configs/task/predict_spcfgfncvm002ema099_local2.yaml
  configs/task/predict_spcfgfncvm002ema0995_local2.yaml
  configs/task/predict_spcfgfncvm002ema0999_local2.yaml
)
raw_dirs=(
  results_local2_fncvm002ema_raw_a105
  results_local2_fncvm002ema099_a105
  results_local2_fncvm002ema0995_a105
  results_local2_fncvm002ema0999_a105
)

for task in "${tasks[@]}"; do
  checkpoint="$(awk '/^load_ckpt:/ {print $2}' "$task")"
  [[ -f "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 2; }
done
for directory in "${raw_dirs[@]}"; do
  [[ ! -e "$directory" ]] || { echo "Refusing to overwrite: $directory" >&2; exit 2; }
done

pids=()
for index in "${!tasks[@]}"; do
  label="${labels[$index]}"
  gpu="${GPUS[$index]}"
  log="/root/log/2026-07-14/ema_predict_${label}.log"
  echo "[predict] $label on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" /root/miniconda3/bin/python run.py \
    --task "${tasks[$index]}" >"$log" 2>&1 &
  pids+=("$!")
done

failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "Prediction failed: ${labels[$index]}" >&2
    failed=1
  fi
done
[[ "$failed" -eq 0 ]] || exit 1

adaptive_dirs=()
for index in "${!raw_dirs[@]}"; do
  raw_dir="${raw_dirs[$index]}"
  adaptive_dir="${raw_dir}_adaptive"
  adaptive_dirs+=("$adaptive_dir")
  /root/miniconda3/bin/python scripts/calibrate_predictions.py \
    --pred-dir "$raw_dir" \
    --noisy-dir . \
    --out-dir "$adaptive_dir" \
    --expected-count 62 \
    --manifest "experiments/spcfgfncvm002ema_spcf/${labels[$index]}_local2_alpha.tsv"
done

/root/miniconda3/bin/python scripts/compare_predictions.py \
  --data-dir localtest2 \
  --reference baseline_adaptive \
  --prediction baseline_adaptive=results_local2_fncvm002a105_adaptive \
  --prediction raw="${raw_dirs[0]}" \
  --prediction raw_adaptive="${adaptive_dirs[0]}" \
  --prediction ema099="${raw_dirs[1]}" \
  --prediction ema099_adaptive="${adaptive_dirs[1]}" \
  --prediction ema0995="${raw_dirs[2]}" \
  --prediction ema0995_adaptive="${adaptive_dirs[2]}" \
  --prediction ema0999="${raw_dirs[3]}" \
  --prediction ema0999_adaptive="${adaptive_dirs[3]}" \
  --workers "${EVAL_WORKERS:-16}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES:-10000}" \
  --output-tsv experiments/spcfgfncvm002ema_spcf/local2_comparison.tsv \
  | tee /root/log/2026-07-14/ema_local2_comparison.log
