#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPU="${GPU:-0}"
TASK_CONFIG="${TASK_CONFIG:-configs/task/predict_spcfgfncvm002a105.yaml}"
OUT_DIR="${OUT_DIR:-submission_results/cvm002a105_spcfgfncvm002a105}"
ZIP_PATH="${ZIP_PATH:-submission_results/result_cvm002a105_spcfgfncvm002a105.zip}"
EXPECTED_COUNT="${EXPECTED_COUNT:-200}"
EXPECTED_POINTS="${EXPECTED_POINTS:-50000}"
SKIP_PREDICT="${SKIP_PREDICT:-0}"
FORCE="${FORCE:-0}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -x /root/miniconda3/bin/python ]]; then
  PYTHON_BIN=/root/miniconda3/bin/python
else
  PYTHON_BIN="$(command -v python)"
fi

export HOME="${HOME_OVERRIDE:-/root/data-tmp/jittor_home_cvm002}"
export TMPDIR="${TMPDIR:-/root/data-tmp/tmp}"
export TMP="${TMP:-$TMPDIR}"
export TEMP="${TEMP:-$TMPDIR}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
mkdir -p "$TMPDIR" submission_results

[[ -f "$TASK_CONFIG" ]] || { echo "Missing task config: $TASK_CONFIG" >&2; exit 2; }

validate_target_path() {
  "$PYTHON_BIN" - "$ROOT" "$1" <<'PY'
import os
import sys

root = os.path.realpath(sys.argv[1])
target = os.path.realpath(os.path.join(root, sys.argv[2]))
if os.path.commonpath([root, target]) != root or target == root:
    raise SystemExit(f"unsafe output path outside repository: {target}")
PY
}

validate_target_path "$OUT_DIR"
validate_target_path "$ZIP_PATH"

if [[ "$SKIP_PREDICT" != "1" ]]; then
  if [[ -e "$OUT_DIR" ]]; then
    if [[ "$FORCE" == "1" ]]; then
      rm -rf -- "$OUT_DIR"
    else
      echo "$OUT_DIR already exists; set FORCE=1 to rebuild it." >&2
      exit 3
    fi
  fi
  CUDA_VISIBLE_DEVICES="$GPU" use_mpi=0 \
    "$PYTHON_BIN" run.py --task "$TASK_CONFIG"
elif [[ ! -d "$OUT_DIR" ]]; then
  echo "SKIP_PREDICT=1 but $OUT_DIR does not exist." >&2
  exit 4
fi

"$PYTHON_BIN" - "$OUT_DIR" "$EXPECTED_COUNT" "$EXPECTED_POINTS" <<'PY'
import sys
from pathlib import Path

import numpy as np

out_dir = Path(sys.argv[1])
expected_count = int(sys.argv[2])
expected_points = int(sys.argv[3])
candidate_roots = [
    out_dir / "shapenet",
    out_dir / "dataset_test_noisy" / "shapenet",
]
populated_roots = [
    root for root in candidate_roots if any(root.glob("*/*/denoised.npy"))
]
if len(populated_roots) != 1:
    raise SystemExit(
        "expected exactly one prediction tree at "
        f"{candidate_roots}, found {len(populated_roots)}"
    )
prediction_root = populated_roots[0]
files = sorted(prediction_root.glob("*/*/denoised.npy"))
if len(files) != expected_count:
    raise SystemExit(f"expected {expected_count} predictions, found {len(files)}")

for path in files:
    points = np.load(path, allow_pickle=False)
    if points.shape != (expected_points, 3):
        raise SystemExit(f"invalid shape {points.shape} in {path}")
    if not np.issubdtype(points.dtype, np.floating):
        raise SystemExit(f"invalid dtype {points.dtype} in {path}")
    if not np.isfinite(points).all():
        raise SystemExit(f"non-finite value in {path}")

print(f"validated {len(files)} predictions with shape ({expected_points}, 3)")
PY

if [[ -e "$ZIP_PATH" ]]; then
  if [[ "$FORCE" == "1" ]]; then
    rm -f -- "$ZIP_PATH" "$ZIP_PATH.sha256"
  else
    echo "$ZIP_PATH already exists; set FORCE=1 to replace it." >&2
    exit 5
  fi
fi

"$PYTHON_BIN" - "$OUT_DIR" "$ZIP_PATH" <<'PY'
import sys
import zipfile
from pathlib import Path

out_dir = Path(sys.argv[1])
zip_path = Path(sys.argv[2])
candidate_roots = [
    out_dir / "shapenet",
    out_dir / "dataset_test_noisy" / "shapenet",
]
populated_roots = [
    root for root in candidate_roots if any(root.glob("*/*/denoised.npy"))
]
if len(populated_roots) != 1:
    raise SystemExit(
        "expected exactly one prediction tree at "
        f"{candidate_roots}, found {len(populated_roots)}"
    )
prediction_root = populated_roots[0]
files = sorted(prediction_root.glob("*/*/denoised.npy"))
with zipfile.ZipFile(
    zip_path,
    mode="w",
    compression=zipfile.ZIP_DEFLATED,
    compresslevel=6,
    allowZip64=True,
) as archive:
    for path in files:
        member = Path("shapenet") / path.relative_to(prediction_root)
        archive.write(path, member.as_posix())

with zipfile.ZipFile(zip_path, mode="r") as archive:
    bad_file = archive.testzip()
    if bad_file is not None:
        raise SystemExit(f"corrupt member in submission archive: {bad_file}")
    if len(archive.namelist()) != len(files):
        raise SystemExit("submission archive member count mismatch")
print(f"validated ZIP with {len(files)} members")
PY

SHA256="$("$PYTHON_BIN" - "$ZIP_PATH" <<'PY'
import hashlib
import sys

digest = hashlib.sha256()
with open(sys.argv[1], "rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
)"
printf '%s  %s\n' "$SHA256" "$ZIP_PATH" | tee "$ZIP_PATH.sha256"

MANIFEST="submission_results/manifest_$(date +%Y%m%d_%H%M%S).tsv"
printf 'task\tout_dir\tzip\tpred_count\tpoints_per_cloud\tsha256\n' > "$MANIFEST"
printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$TASK_CONFIG" "$OUT_DIR" "$ZIP_PATH" "$EXPECTED_COUNT" "$EXPECTED_POINTS" \
  "$SHA256" >> "$MANIFEST"

echo "Created $ZIP_PATH"
echo "Manifest: $MANIFEST"
