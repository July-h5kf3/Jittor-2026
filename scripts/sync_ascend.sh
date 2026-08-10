#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
test "$(git -C "$ROOT" branch --show-current)" = new
rsync -az --delete \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.venv' \
    --exclude 'train_logs' \
    --exclude 'checkpoints' \
    --exclude 'predictions' \
    "$ROOT/" zhiyuan-huawei:/data/ldc/Track2-new/
