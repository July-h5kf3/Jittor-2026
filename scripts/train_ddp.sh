#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

NP="${NP:-}"
TASK="${1:-configs/task/train_vm.yaml}"
shift || true

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif [[ -x /root/miniconda3/bin/python ]]; then
    PYTHON_BIN="/root/miniconda3/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "Python interpreter not found; set PYTHON_BIN explicitly." >&2
    exit 1
  fi
fi

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
export JITTOR_CACHE_PER_RANK="${JITTOR_CACHE_PER_RANK:-0}"

# OpenMPI stores its shared-memory coordination file below the session
# directory.  On the training container /tmp belongs to the nearly-full root
# filesystem, while /dev/shm has ample capacity and is the correct backing
# store for same-node collectives.  Keep MPI scratch off the root filesystem
# for both ordinary and per-rank-cache jobs.
export OMPI_MCA_orte_tmpdir_base="${OMPI_MCA_orte_tmpdir_base:-/dev/shm/jittor-openmpi-${USER:-user}}"
export OMPI_MCA_btl_vader_backing_directory="${OMPI_MCA_btl_vader_backing_directory:-/dev/shm}"
mkdir -p "$OMPI_MCA_orte_tmpdir_base"

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

if [[ "$JITTOR_CACHE_PER_RANK" == "1" ]]; then
  # A newly introduced Jittor operator may otherwise be compiled by every MPI
  # rank into the same cache at once. That race can corrupt generated objects
  # or crash the compiler runtime. Keep both cache and compiler scratch local
  # to each rank for first-run/experimental jobs. Jittor otherwise starts up
  # to 16 compiler workers inside every rank and also enables its C++ parallel
  # op compiler; that nested concurrency has produced heap corruption while
  # compiling a newly introduced loss. Compile serially during cache warmup.
  export DISABLE_MULTIPROCESSING="${DISABLE_MULTIPROCESSING:-1}"
  export use_parallel_op_compiler="${use_parallel_op_compiler:-0}"
  export JITTOR_CACHE_BASE="$cache_path"
  export JITTOR_TMP_BASE="$TMPDIR"
  export PYTHON_BIN TASK
  exec "$mpirun_bin" \
    --allow-run-as-root \
    --bind-to none \
    --map-by slot \
    -x WANDB_MODE -x WANDB_PROJECT -x WANDB_DIR -x WANDB_CONSOLE \
    -x OMPI_MCA_orte_tmpdir_base -x OMPI_MCA_btl_vader_backing_directory \
    -x DISABLE_MULTIPROCESSING -x use_parallel_op_compiler \
    -x JITTOR_CACHE_BASE -x JITTOR_TMP_BASE -x PYTHON_BIN -x TASK \
    -np "$NP" \
    /bin/bash -c '
      set -euo pipefail
      rank="${OMPI_COMM_WORLD_LOCAL_RANK:-${OMPI_COMM_WORLD_RANK:-0}}"
      export cache_path="${JITTOR_CACHE_BASE}/rank${rank}"
      export TMPDIR="${JITTOR_TMP_BASE}/rank${rank}"
      export TMP="$TMPDIR" TEMP="$TMPDIR"
      mkdir -p "$cache_path" "$TMPDIR"
      exec "$PYTHON_BIN" run.py --task "$TASK" "$@"
    ' _ "$@"
fi

exec "$mpirun_bin" \
  --allow-run-as-root \
  --bind-to none \
  --map-by slot \
  -x WANDB_MODE -x WANDB_PROJECT -x WANDB_DIR -x WANDB_CONSOLE \
  -x OMPI_MCA_orte_tmpdir_base -x OMPI_MCA_btl_vader_backing_directory \
  -x TMPDIR -x TMP -x TEMP -x cache_path \
  -np "$NP" \
  "$PYTHON_BIN" run.py --task "$TASK" "$@"
