#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
if [ -z "${INPUT_ROOT:-}" ]; then
  echo "set INPUT_ROOT to the noisy test tree" >&2
  exit 2
fi
py=${PYTHON:-python}
"$py" -X utf8 "$root/fusion/fuse_global_m3i1_vf_n01.py" \
  --noisy-root "$INPUT_ROOT" \
  --key-list "$root/test_200_keys.txt" \
  --expected-count 200 \
  --output-root "$root/preds/fused_global_m3i1_vf_n01" \
  --mamba-48k "$root/preds/mamba_mix75_s48000/merged" \
  --mamba-60k "$root/preds/mamba_v8192_s060000/merged" \
  --ipfn-1024 "$root/preds/ipfn_1024" \
  --vm-fixed "$root/preds/vm_fixed"
"$py" -X utf8 "$root/code/package_result.py" \
  --pred-root "$root/preds/fused_global_m3i1_vf_n01" \
  --zip-path "$root/result.zip" \
  --audit-json "$root/result_audit.json"
echo FUSED_AND_PACKED "$root/result.zip"
