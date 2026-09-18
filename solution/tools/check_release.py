"""CPU preflight for the frozen 200-item competition inference pipeline."""
import argparse
import json
from pathlib import Path
import numpy as np
from prepare_keys import discover
from download_weights import digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root', type=Path, required=True)
    args = p.parse_args()
    observed = discover(args.input_root, 200)
    expected = [f'shapenet/00000000/btest_{i:06d}' for i in range(1, 201)]
    if observed != expected:
        raise ValueError('The frozen result packager expects btest_000001 ... btest_000200; see docs/REPRODUCE.md for custom data.')
    keyfile = ROOT / 'test_200_keys.txt'
    if not keyfile.is_file() or sorted(keyfile.read_text(encoding='utf8').splitlines()) != observed:
        raise ValueError('Run tools/prepare_keys.py on this input tree first.')
    base = args.input_root / 'shapenet' if (args.input_root / 'shapenet').is_dir() else args.input_root
    for key in observed:
        value = np.load(base / key.removeprefix('shapenet/') / 'noisy.npy', mmap_mode='r', allow_pickle=False)
        if value.shape != (50000, 3) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f'Invalid noisy input: {key}')
    manifest = json.loads((ROOT / 'weights/manifest.json').read_text(encoding='utf8'))
    for item in manifest['files']:
        path = ROOT / item['path']
        if not path.is_file() or path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
            raise ValueError(f'Missing or altered checkpoint: {path}; run tools/download_weights.py')
    print('Preflight passed: 200 real inputs, 50000 points each, four verified checkpoints.')


if __name__ == '__main__':
    main()
