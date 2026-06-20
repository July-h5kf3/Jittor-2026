#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

NP="${NP:-}"
TASK="${1:-configs/task/train_vm.yaml}"
shift || true

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
    if [[ "$gpu_count" -gt 0 ]]; then
      CUDA_VISIBLE_DEVICES="$(seq -s, 0 "$((gpu_count - 1))")"
    else
      CUDA_VISIBLE_DEVICES="0"
    fi
  else
    CUDA_VISIBLE_DEVICES="0"
  fi
fi

if [[ -z "$NP" ]]; then
  IFS=',' read -r -a visible_devices <<< "$CUDA_VISIBLE_DEVICES"
  NP="${#visible_devices[@]}"
fi

export CUDA_VISIBLE_DEVICES
export use_mpi="${use_mpi:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
if [[ -d /root/data-tmp ]]; then
  export cache_path="${cache_path:-/root/data-tmp/.cache/jittor}"
fi

if [[ -n "${MPIRUN:-}" ]]; then
  mpirun_bin="$MPIRUN"
elif command -v mpirun >/dev/null 2>&1; then
  mpirun_bin="$(command -v mpirun)"
else
  mpirun_bin="$(python - <<'PY'
import jittor as jt
mpicc = getattr(jt.compile_extern, "mpicc_path", "")
print(mpicc.replace("mpicc", "mpirun") if mpicc else "")
PY
)"
fi

if [[ -z "$mpirun_bin" || ! -x "$mpirun_bin" ]]; then
  echo "mpirun not found. Install OpenMPI in the active environment first." >&2
  exit 1
fi

exec "$mpirun_bin" \
  --allow-run-as-root \
  --bind-to none \
  --map-by slot \
  -np "$NP" \
  python run.py --task "$TASK" "$@"
