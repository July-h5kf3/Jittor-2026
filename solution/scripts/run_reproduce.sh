#!/usr/bin/env bash
set -euo pipefail
# Reproduce B-board 76.999: four member infers then median3 - 0.01 VM residual.
root=$(cd "$(dirname "$0")/.." && pwd)
if [ -z "${INPUT_ROOT:-}" ]; then
  echo "set INPUT_ROOT to the official test noisy tree (contains shapenet/.../noisy.npy)" >&2
  exit 2
fi
export PYTHON="${PYTHON:-python}"
"$PYTHON" "$root/tools/check_release.py" --input-root "$INPUT_ROOT"
world_mamba=${WORLD_MAMBA:-1}
world_ipfn=${WORLD_IPFN:-1}
world_vm=${WORLD_VM:-1}
bash "$root/scripts/infer_mamba.sh" 48k "$world_mamba"
bash "$root/scripts/infer_mamba.sh" 60k "$world_mamba"
bash "$root/scripts/infer_ipfn.sh" 1024 "$world_ipfn"
bash "$root/scripts/infer_vm.sh" "$world_vm"
bash "$root/scripts/fuse.sh"
echo REPRODUCE_DONE
