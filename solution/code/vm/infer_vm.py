#!/usr/bin/env python3
"""Jittor VM inference for the B-board fixed-v8192 member."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jittor as jt
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from model_vm import EMBEDDING_DIM, VelocityModule  # noqa: E402


def fps(points: np.ndarray, count: int) -> np.ndarray:
    ids = np.empty(count, np.int64)
    ids[0] = 0
    dist = np.full(len(points), np.inf)
    for i in range(count):
        if i:
            ids[i] = int(np.argmax(dist))
        delta = points - points[ids[i]]
        dist = np.minimum(dist, np.einsum("ij,ij->i", delta, delta))
    return ids


def infer_one(model, noisy):
    count = len(noisy)
    patch_size = 1000
    patches = max(1, int(6 * count / patch_size))
    tree = cKDTree(noisy)
    centers = fps(noisy, patches)
    rows, positions, distances = [], [], []
    for center in centers:
        dist, idx = tree.query(noisy[center], k=patch_size)
        seed = noisy[center]
        rows.append(noisy[idx] - seed)
        positions.append(idx)
        distances.append(dist / (dist[-1] + 1e-8))
    arr = np.asarray(rows, np.float32)
    outs = []
    with jt.no_grad():
        for start in range(0, patches, 2):
            tensor = jt.array(arr[start : start + 2])
            for _ in range(4):
                feat = model.encoder(tensor)
                pred = model.decoder(feat.reshape(-1, EMBEDDING_DIM)).reshape(
                    tensor.shape[0], tensor.shape[1], 3
                )
                tensor = tensor + pred / 4.0
            outs.append(tensor.numpy())
    pred = np.concatenate(outs, 0)
    result = noisy.copy()
    num = np.zeros_like(result)
    den = np.zeros(count, np.float32)
    for patch, (idx, dist) in enumerate(zip(positions, distances)):
        weight = np.exp(-(dist - dist.min()) / 0.10).astype(np.float32)
        num[idx] += pred[patch] * weight[:, None]
        den[idx] += weight
    covered = den > 0
    result[covered] = num[covered] / den[covered, None]
    return result


def load_keys(path: Path) -> list[str]:
    return [
        line.strip().replace("\\", "/").strip("/")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def noisy_path(root: Path, key: str) -> Path:
    options = [
        root.joinpath(*key.split("/"), "noisy.npy"),
        root.joinpath("shapenet", *key.split("/"), "noisy.npy"),
        root.joinpath("dataset_test_noisy", *key.split("/"), "noisy.npy"),
    ]
    for path in options:
        if path.is_file():
            return path
    raise FileNotFoundError("%s / %s" % (root, key))


def load_weights(model, checkpoint: Path) -> None:
    if checkpoint.suffix != ".npz":
        raise ValueError("VM checkpoint must be a Jittor npz: %s" % checkpoint)
    with np.load(checkpoint, allow_pickle=False) as archive:
        arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    arrays = {
        name: value
        for name, value in arrays.items()
        if not name.endswith("num_batches_tracked")
    }
    expected = model.state_dict()
    missing = [name for name in expected if name not in arrays]
    if missing:
        raise ValueError("missing VM parameters: %s" % missing[:8])
    matched = {name: arrays[name] for name in expected}
    for name, value in matched.items():
        if tuple(expected[name].shape) != tuple(value.shape):
            raise ValueError(
                "shape mismatch %s: model %s vs ckpt %s"
                % (name, tuple(expected[name].shape), tuple(value.shape))
            )
    model.load_parameters(matched)
    for name, module in (
        ("decoder.bn_1_out", model.decoder.bn_1_out),
        ("decoder.bn_2_out", model.decoder.bn_2_out),
    ):
        if hasattr(module, "running_mean") and name + ".running_mean" in arrays:
            module.running_mean.assign(arrays[name + ".running_mean"])
            module.running_var.assign(arrays[name + ".running_var"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--key-list", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    jt.flags.use_cuda = 1 if args.device == "cuda" else 0
    keys = load_keys(args.key_list)
    keys = keys[args.rank :: args.world_size]
    model = VelocityModule()
    load_weights(model, args.checkpoint)
    model.eval()
    args.output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for index, key in enumerate(keys, start=1):
        noisy = np.load(noisy_path(args.input_root, key)).astype(np.float32)
        pred = infer_one(model, noisy).astype(np.float32)
        dest_key = key if key.startswith("shapenet/") else "shapenet/" + key
        out = args.output_root.joinpath(*dest_key.split("/"), "denoised.npy")
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, pred)
        if index % 10 == 0 or index == len(keys):
            print(json.dumps({"done": index, "total": len(keys), "elapsed": time.time() - started}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
