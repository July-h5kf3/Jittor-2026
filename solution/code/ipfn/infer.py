"""Competition-tree inference for the Jittor IterativePFN port."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jittor as jt
import numpy as np
from scipy.spatial import cKDTree

from model import IterativePFN, load_converted_weights


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=1000)
    parser.add_argument("--seed-k", type=int, default=6)
    parser.add_argument("--seed-k-alpha", type=int, default=10)
    parser.add_argument("--num-modules", type=int, default=4)
    parser.add_argument("--blend-alpha", type=float, default=0.0)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--merge-mode", choices=("nearest", "softmax", "softmax010"), default="softmax010")
    parser.add_argument("--merge-sigma", type=float, default=0.1)
    return parser.parse_args()


def resolve_shapenet(path):
    root = Path(path).expanduser().resolve()
    return root / "shapenet" if (root / "shapenet").is_dir() else root


def normalize(points):
    center = 0.5 * (points.max(axis=0, keepdims=True) + points.min(axis=0, keepdims=True))
    centered = points - center
    scale = np.linalg.norm(centered, axis=1).max()
    return centered / max(float(scale), 1e-12), center, np.float32(scale)


def farthest_indices(points, count):
    count = min(int(count), len(points))
    selected = np.empty(count, dtype=np.int64)
    min_dist = np.full(len(points), np.inf, dtype=np.float32)
    farthest = 0
    for index in range(count):
        selected[index] = farthest
        dist = np.square(points - points[farthest]).sum(axis=1)
        min_dist = np.minimum(min_dist, dist)
        farthest = int(np.argmax(min_dist))
    return selected


def patch_denoise(model, points, patch_size, seed_k, seed_k_alpha, num_modules,
                  merge_mode="nearest", merge_sigma=0.1):
    count = len(points)
    num_patches = max(1, int(seed_k * count / patch_size))
    seed_idx = farthest_indices(points, num_patches)
    seeds = points[seed_idx]
    tree = cKDTree(points)
    dists, indices = tree.query(seeds, k=patch_size, workers=-1)
    dists = np.square(np.asarray(dists, dtype=np.float32))
    indices = np.asarray(indices, dtype=np.int64)
    patches = points[indices] - seeds[:, None, :]
    normalized_dists = dists / np.maximum(dists[:, -1:], 1e-12)

    patch_step = max(1, int(count / max(1, seed_k_alpha * patch_size)))
    predictions = []
    for start in range(0, num_patches, patch_step):
        batch = jt.array(np.ascontiguousarray(patches[start : start + patch_step], dtype=np.float32))
        predictions.append(model(batch, num_modules=num_modules).numpy())
    predictions = np.concatenate(predictions, axis=0) + seeds[:, None, :]

    if merge_mode == "nearest":
        best_dist = np.full(count, np.inf, dtype=np.float32)
        output = points.copy()
        for patch_id in range(num_patches):
            point_idx = indices[patch_id]
            candidate_dist = normalized_dists[patch_id]
            better = candidate_dist < best_dist[point_idx]
            chosen = point_idx[better]
            output[chosen] = predictions[patch_id, better]
            best_dist[chosen] = candidate_dist[better]
    else:
        # softmax010: exp(-(d - d_min) / T) with T=0.10.
        # d_min is the smallest covering-patch distance of each global point.
        temperature = 0.10 if merge_mode == "softmax010" else max(float(merge_sigma), 1e-6)
        d_min = np.full(count, np.inf, dtype=np.float32)
        for patch_id in range(num_patches):
            point_idx = indices[patch_id]
            d_min[point_idx] = np.minimum(d_min[point_idx], normalized_dists[patch_id])
        output_sum = np.zeros((count, 3), dtype=np.float64)
        weight_sum = np.zeros(count, dtype=np.float64)
        for patch_id in range(num_patches):
            point_idx = indices[patch_id]
            weights = np.exp(
                -(normalized_dists[patch_id] - d_min[point_idx]) / temperature
            ).astype(np.float64)
            np.add.at(output_sum, point_idx, predictions[patch_id] * weights[:, None])
            np.add.at(weight_sum, point_idx, weights)
        output = points.copy()
        covered = weight_sum > 0
        output[covered] = (output_sum[covered] / weight_sum[covered, None]).astype(np.float32)
    return np.asarray(output, dtype=np.float32)


def main():
    args = parse_args()
    jt.flags.use_cuda = 1
    jt.flags.use_tensorcore = 1
    model = load_converted_weights(IterativePFN(num_modules=4), args.weights)
    input_root = resolve_shapenet(args.input_root)
    output_root = Path(args.output_root).expanduser().resolve()
    paths = sorted(input_root.glob("*/*/noisy.npy"))
    if args.limit > 0:
        paths = paths[: args.limit]
    paths = paths[args.rank :: args.world_size]
    started = time.time()
    processed = 0
    for index, path in enumerate(paths, 1):
        key = path.parent.relative_to(input_root)
        save = output_root / "shapenet" / key / "denoised.npy"
        noisy = np.load(path).astype(np.float32, copy=False)
        if args.skip_existing and save.is_file():
            value = np.load(save, mmap_mode="r")
            if value.shape == noisy.shape and value.dtype == np.float32:
                continue
        if args.no_normalize:
            normalized = noisy
            center = np.zeros((1, 3), dtype=np.float32)
            scale = np.float32(1.0)
        else:
            normalized, center, scale = normalize(noisy)
        prediction = patch_denoise(
            model,
            normalized.astype(np.float32),
            args.patch_size,
            args.seed_k,
            args.seed_k_alpha,
            args.num_modules,
            args.merge_mode,
            args.merge_sigma,
        )
        prediction = prediction * scale + center
        if args.blend_alpha:
            prediction = prediction + np.float32(args.blend_alpha) * (noisy - prediction)
        prediction = np.asarray(prediction, dtype=np.float32)
        if prediction.shape != noisy.shape or not np.isfinite(prediction).all():
            raise RuntimeError("invalid output for %s" % key)
        save.parent.mkdir(parents=True, exist_ok=True)
        np.save(save, prediction)
        processed += 1
        print(
            "[rank %d %d/%d] %s %.2fs/case"
            % (args.rank, index, len(paths), key.as_posix(), (time.time() - started) / max(1, processed)),
            flush=True,
        )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / ("jittor_summary_rank%02d.json" % args.rank)).write_text(
        json.dumps({"processed": processed, "elapsed_seconds": time.time() - started}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
