#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
if [ -z "${INPUT_ROOT:-}" ]; then
  echo "set INPUT_ROOT to the noisy test tree" >&2
  exit 2
fi
py=${PYTHON:-python}
world=${1:-1}
mkdir -p "$root/preds/vm_fixed" "$root/logs"
rank=0
pids=()
while [ "$rank" -lt "$world" ]; do
  env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$rank}" \
    PYTHONPATH="$root/code/vm" \
    "$py" -X utf8 "$root/code/vm/infer_vm.py" \
      --checkpoint "$root/weights/vm_fixed/fixed_v8192_epoch10.npz" \
      --input-root "$INPUT_ROOT" \
      --output-root "$root/preds/vm_fixed" \
      --key-list "$root/test_200_keys.txt" \
      --rank "$rank" --world-size "$world" --device cuda \
      >"$root/logs/vm_fixed_r${rank}.log" 2>&1 &
  pids+=("$!")
  echo started vm_fixed rank=$rank pid=$!
  rank=$((rank + 1))
done
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
if [ "$fail" -ne 0 ]; then
  echo "inference worker failed; see logs" >&2
  exit 1
fi
echo VM_DONE fail=$fail
