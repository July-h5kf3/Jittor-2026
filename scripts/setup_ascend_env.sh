#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/data/ldc/Track2-new
ENV_ROOT=/data/ldc/envs/track2-ascend
WHEEL_ROOT=/data/ldc/packages/track2-ascend
JITTOR_SHA=06f5d3d271555682c95aa3505518f47eeab2bd9c
JITTOR_ROOT=/data/ldc/vendor/jittor-06f5d3d271555682c95aa3505518f47eeab2bd9c

test -f "$PROJECT_ROOT/requirements-ascend.txt"
test -f "$JITTOR_ROOT/setup.py"
test -d "$WHEEL_ROOT"
mkdir -p /data/ldc/envs /data/ldc/cache/jittor-track2-ascend
if [ ! -x "$ENV_ROOT/bin/python" ]; then
    python -m venv --system-site-packages "$ENV_ROOT"
fi
"$ENV_ROOT/bin/python" -m pip install \
    --no-index --find-links "$WHEEL_ROOT" \
    -r "$PROJECT_ROOT/requirements-ascend.txt"
"$ENV_ROOT/bin/python" -m pip install \
    --no-index --no-build-isolation --no-deps -e "$JITTOR_ROOT"
