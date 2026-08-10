#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/data/ldc/Track2-new
ENV_ROOT=/data/ldc/envs/track2-ascend
WHEEL_ROOT=/data/ldc/packages/track2-ascend
JITTOR_SHA=06f5d3d271555682c95aa3505518f47eeab2bd9c
JITTOR_ROOT=/data/ldc/vendor/jittor-06f5d3d271555682c95aa3505518f47eeab2bd9c

if [ ! -f "$PROJECT_ROOT/requirements-ascend.txt" ]; then
    printf 'error: requirements file not found: %s\n' "$PROJECT_ROOT/requirements-ascend.txt" >&2
    exit 1
fi
if [ ! -f "$JITTOR_ROOT/setup.py" ]; then
    printf 'error: Jittor setup.py not found: %s\n' "$JITTOR_ROOT/setup.py" >&2
    exit 1
fi
if [ ! -d "$WHEEL_ROOT" ]; then
    printf 'error: wheel directory not found: %s\n' "$WHEEL_ROOT" >&2
    exit 1
fi
if ! actual_jittor_root=$(git -c safe.directory="$JITTOR_ROOT" -C "$JITTOR_ROOT" rev-parse --show-toplevel); then
    printf 'error: unable to read Jittor repository root from %s\n' "$JITTOR_ROOT" >&2
    exit 1
fi
if [ "$actual_jittor_root" != "$JITTOR_ROOT" ]; then
    printf 'error: Jittor repository root is %s; expected %s\n' "$actual_jittor_root" "$JITTOR_ROOT" >&2
    exit 1
fi
if ! actual_jittor_sha=$(git -c safe.directory="$JITTOR_ROOT" -C "$JITTOR_ROOT" rev-parse HEAD); then
    printf 'error: unable to read Jittor revision from %s\n' "$JITTOR_ROOT" >&2
    exit 1
fi
if [ "$actual_jittor_sha" != "$JITTOR_SHA" ]; then
    printf 'error: Jittor revision is %s; expected %s\n' "$actual_jittor_sha" "$JITTOR_SHA" >&2
    exit 1
fi
if ! jittor_status=$(git -c safe.directory="$JITTOR_ROOT" -C "$JITTOR_ROOT" status --porcelain); then
    printf 'error: unable to inspect Jittor working tree: %s\n' "$JITTOR_ROOT" >&2
    exit 1
fi
if [ -n "$jittor_status" ]; then
    printf 'error: Jittor working tree is not clean: %s\n' "$JITTOR_ROOT" >&2
    exit 1
fi
chown -R "$(id -u):$(id -g)" "$JITTOR_ROOT"
mkdir -p /data/ldc/envs /data/ldc/cache/jittor-track2-ascend
if [ ! -x "$ENV_ROOT/bin/python" ]; then
    python3.11 -m venv --system-site-packages "$ENV_ROOT"
fi
"$ENV_ROOT/bin/python" - <<'PY'
import platform
import sys

if sys.version_info[:2] != (3, 11):
    raise SystemExit(f"error: expected Python 3.11, got {sys.version}")
if platform.machine().lower() != "aarch64":
    raise SystemExit(f"error: expected aarch64, got {platform.machine()}")
PY
"$ENV_ROOT/bin/python" -m pip install \
    --no-index --find-links "$WHEEL_ROOT" \
    -r "$PROJECT_ROOT/requirements-ascend.txt"
"$ENV_ROOT/bin/python" -m pip install \
    --no-index --no-build-isolation --no-deps -e "$JITTOR_ROOT"
