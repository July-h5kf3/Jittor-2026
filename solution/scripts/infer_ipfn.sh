#!/usr/bin/env bash
set -euo pipefail
# usage: infer_ipfn.sh 1024|8192 WORLD
which=${1:?1024 or 8192}
world=${2:-8}
root=$(cd "$(dirname "$0")/.." && pwd)
if [ -z "${INPUT_ROOT:-}" ]; then
  echo "set INPUT_ROOT to the noisy test tree" >&2
  exit 2
fi
py=${PYTHON:-python}
case "$which" in
  1024)
    weights=$root/weights/ipfn_1024/denoisenet-jittor-b-epoch10.npz
    name=ipfn_1024
    ;;
  8192)
    weights=$root/weights/ipfn_8192/denoisenet-jittor-b-epoch10.npz
    name=ipfn_8192
    ;;
  *) echo "which must be 1024 or 8192" >&2; exit 2 ;;
esac
mkdir -p "$root/preds/$name" "$root/jittor_home" "$root/logs"
rank=0
pids=()
while [ "$rank" -lt "$world" ]; do
  env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$rank}" \
    JITTOR_HOME=$root/jittor_home/${name}_g${rank} \
    PYTHONPATH="$root/code/ipfn" \
    "$py" -X utf8 "$root/code/ipfn/infer.py" \
      --weights "$weights" --input-root "$INPUT_ROOT" \
      --output-root "$root/preds/$name" \
      --rank "$rank" --world-size "$world" \
      --patch-size 2000 --seed-k 6 --seed-k-alpha 10 --num-modules 4 \
      --no-normalize --merge-mode softmax010 --blend-alpha 0 \
      >"$root/logs/${name}_r${rank}.log" 2>&1 &
  pids+=("$!")
  echo started $name rank=$rank pid=$!
  rank=$((rank + 1))
done
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
if [ "$fail" -ne 0 ]; then
  echo "inference worker failed; see logs" >&2
  exit 1
fi
echo IPFN_DONE $name fail=$fail
