#!/usr/bin/env bash
set -euo pipefail
# usage: infer_mamba.sh 48k|60k WORLD
which=${1:?48k or 60k}
world=${2:-7}
root=$(cd "$(dirname "$0")/.." && pwd)
if [ -z "${INPUT_ROOT:-}" ]; then
  echo "set INPUT_ROOT to the noisy test tree" >&2
  exit 2
fi
py=${PYTHON:-python}
case "$which" in
  48k)
    ckpt=$root/weights/mamba_mix75_s48000/step-048000-jittor.pkl
    name=mamba_mix75_s48000
    ;;
  60k)
    ckpt=$root/weights/mamba_v8192_s060000/step-060000-jittor.pkl
    name=mamba_v8192_s060000
    ;;
  *) echo "which must be 48k or 60k" >&2; exit 2 ;;
esac
code=$root/code/mamba
out=$root/preds/$name
shards=$out/shards
mkdir -p "$shards" "$root/jittor_home" "$root/logs"
rank=0
pids=()
while [ "$rank" -lt "$world" ]; do
  shard=$shards/shard-$rank-of-$world
  home=$root/jittor_home/${name}_g${rank}
  mkdir -p "$home"
  rm -rf "$shard"
  env HOME="$home" JITTOR_HOME="$home" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$rank}" \
    PYTHONPATH="$code/jittor_core:$code" \
    "$py" -X utf8 "$code/run_inference_shard.py" \
      --code-root "$code/jittor_core" \
      --inference-script "$code/jittor_core/infer.py" \
      --checkpoint "$ckpt" --input-root "$INPUT_ROOT" \
      --key-list "$root/test_200_keys.txt" --output-root "$shard" \
      --shard-index "$rank" --shard-count "$world" \
      --patch-size 2000 --seed-k 6 --seed-k-alpha 20 \
      --merge-strategy softmax010 --softmax-temperature 0.10 \
      --seed 2020 --device cuda \
      >"$root/logs/${name}_g${rank}.log" 2>&1 &
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
args=()
rank=0
while [ "$rank" -lt "$world" ]; do
  args+=(--shard "$shards/shard-$rank-of-$world")
  rank=$((rank + 1))
done
PYTHONPATH="$code" "$py" -X utf8 "$code/merge_inference_shards.py" \
  "${args[@]}" --expected-key-list "$root/test_200_keys.txt" --output-root "$out/merged"
echo MERGED $name fail=$fail
