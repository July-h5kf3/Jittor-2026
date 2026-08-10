#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT=${1:?usage: prepare_ascend_bundle.sh BUNDLE_ROOT}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
JITTOR_SHA=06f5d3d271555682c95aa3505518f47eeab2bd9c
JITTOR_ROOT="$BUNDLE_ROOT/jittor-$JITTOR_SHA"
WHEEL_ROOT="$BUNDLE_ROOT/wheels"

test ! -e "$JITTOR_ROOT"
mkdir -p "$BUNDLE_ROOT" "$WHEEL_ROOT"
git clone --filter=blob:none https://github.com/Jittor/jittor.git "$JITTOR_ROOT"
git -C "$JITTOR_ROOT" checkout "$JITTOR_SHA"
test "$(git -C "$JITTOR_ROOT" rev-parse HEAD)" = "$JITTOR_SHA"
python3 -m pip download \
    --only-binary=:all: \
    --platform manylinux2014_aarch64 \
    --python-version 311 \
    --implementation cp \
    --dest "$WHEEL_ROOT" \
    -r "$ROOT/requirements-ascend.txt"
