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
  # The container rootfs (/) is a shared overlay that runs ~full (~171M free).
  # nvcc/gcc write compile intermediates to $TMPDIR (default /tmp, on /), so a
  # heavy new-op compile (e.g. the SPCF stage's distance module) fails with
  # "No space left on device" -- this killed SPCF on 2026-06-29. Redirect all
  # compiler/python scratch to the roomy NFS mount so / is never touched.
  export TMPDIR="${TMPDIR:-/root/data-tmp/Track2/.nvcc_tmp}"
  export TMP="$TMPDIR"
  export TEMP="$TMPDIR"
  mkdir -p "$TMPDIR"
fi

# --- Weights & Biases logging ---
# The GPU box is air-gapped behind a captive portal, so default to OFFLINE:
# runs are written under $WANDB_DIR/wandb/ and synced later from a machine with
# internet (`wandb sync <dir>`). Override WANDB_MODE=online/disabled to change.
# WANDB_CONSOLE=off keeps stdout clean so runners can still grep training logs.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-Track2}"
export WANDB_DIR="${WANDB_DIR:-$(pwd)}"
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"

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
  -x WANDB_MODE -x WANDB_PROJECT -x WANDB_DIR -x WANDB_CONSOLE \
  -x TMPDIR -x TMP -x TEMP -x cache_path \
  -np "$NP" \
  python run.py --task "$TASK" "$@"
