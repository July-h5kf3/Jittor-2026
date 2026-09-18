#!/usr/bin/env python3
"""Jittor-native 3DMambaIPF inference on canonical Track2 NPY trees."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Optional, Sequence, Tuple

import jittor as jt
import numpy as np

from plr3d.io import (
    PLRError,
    discover_clouds,
    load_cloud,
    load_model_checkpoint,
    normalize_unit_sphere,
    save_cloud_new,
    sha256_file,
    write_json_new,
)
from plr3d.models import DenoiseNet


def cyclic_permutation(count: int, requested_shift: int):
    effective = int(requested_shift) % count
    if effective == 0:
        return None, 0
    return np.roll(np.arange(count, dtype=np.int64), -effective), effective


def restore_order(array: np.ndarray, permutation):
    if permutation is None:
        return np.ascontiguousarray(array)
    restored = np.empty_like(array)
    restored[permutation] = array
    return np.ascontiguousarray(restored)


def displacement_summary(noisy: np.ndarray, denoised: np.ndarray):
    norms = np.linalg.norm(
        denoised.astype(np.float64) - noisy.astype(np.float64), axis=1
    )
    return {
        "max": float(norms.max()),
        "mean": float(norms.mean()),
        "median": float(np.median(norms)),
        "p95": float(np.quantile(norms, 0.95)),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--key-list", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--expected-points", type=int, default=50000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--patch-size", type=int, default=2000)
    parser.add_argument("--seed-k", type=int, default=6)
    parser.add_argument("--seed-k-alpha", type=int, default=20)
    parser.add_argument("--num-modules", type=int, default=4)
    parser.add_argument("--frame-knn", type=int, default=32)
    parser.add_argument("--noise-decay", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument(
        "--merge-strategy",
        choices=("softmax010", "best_patch"),
        default="softmax010",
    )
    parser.add_argument("--softmax-temperature", type=float, default=0.10)
    parser.add_argument("--input-cyclic-shift", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    jt.flags.use_cuda = 1 if args.device == "cuda" else 0
    random.seed(args.seed)
    np.random.seed(args.seed)
    jt.set_global_seed(args.seed)

    checkpoint = args.checkpoint.expanduser().resolve()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else output_root / "inference_manifest.json"
    )
    if not checkpoint.is_file() or not input_root.is_dir():
        raise PLRError("checkpoint or input root is missing")
    if output_root.exists() or manifest_path.exists():
        raise PLRError("output root or inference manifest already exists")

    records = discover_clouds(
        input_root,
        "noisy.npy",
        args.key_list.expanduser().resolve() if args.key_list is not None else None,
    )
    if args.limit is not None:
        if args.limit <= 0:
            raise PLRError("--limit must be positive")
        records = records[: args.limit]
    if args.expected_count is not None and len(records) != args.expected_count:
        raise PLRError(
            f"expected {args.expected_count} inputs, discovered {len(records)}"
        )

    model = DenoiseNet(
        frame_knn=args.frame_knn,
        num_modules=args.num_modules,
        noise_decay=args.noise_decay,
    )
    checkpoint_metadata = load_model_checkpoint(model, checkpoint)
    model.eval()

    entries = []
    started = time.time()
    for index, (key, input_path) in enumerate(records, start=1):
        noisy = load_cloud(input_path, expected_points=args.expected_points)
        permutation, effective_shift = cyclic_permutation(
            noisy.shape[0], args.input_cyclic_shift
        )
        model_input = noisy if permutation is None else np.ascontiguousarray(noisy[permutation])
        points = jt.array(model_input).float32()
        normalized, center, scale = normalize_unit_sphere(points)

        cloud_started = time.time()
        prediction = model.patch_based_denoise(
            normalized,
            patch_size=args.patch_size,
            seed_k=args.seed_k,
            seed_k_alpha=args.seed_k_alpha,
            num_modules_to_use=args.num_modules,
            merge_strategy=args.merge_strategy,
            softmax_temperature=args.softmax_temperature,
        )
        prediction = prediction * scale + center
        denoised_permuted = np.ascontiguousarray(prediction.numpy(), dtype=np.float32)
        if denoised_permuted.shape != model_input.shape:
            raise PLRError(
                f"{key} changed shape {model_input.shape} -> {denoised_permuted.shape}"
            )
        denoised = restore_order(denoised_permuted, permutation)
        fallback_indices = list(model.last_patch_stitching_fallback_indices)
        if permutation is not None:
            fallback_indices = sorted(int(permutation[i]) for i in fallback_indices)
        if fallback_indices:
            denoised[np.asarray(fallback_indices, dtype=np.int64)] = noisy[
                np.asarray(fallback_indices, dtype=np.int64)
            ]
        if denoised.shape != noisy.shape or not np.isfinite(denoised).all():
            raise PLRError(f"{key} produced invalid output")

        output_path = output_root.joinpath(*key.split("/"), "denoised.npy")
        save_cloud_new(output_path, denoised)
        entries.append(
            {
                "key": key,
                "input_file_sha256": sha256_file(input_path),
                "output_file_sha256": sha256_file(output_path),
                "output_array_sha256": hashlib.sha256(denoised.tobytes()).hexdigest(),
                "shape": list(denoised.shape),
                "input_cyclic_shift_effective": effective_shift,
                "fps_first_canonical_index": (
                    0 if permutation is None else int(permutation[0])
                ),
                "stitching_fallback_count": len(fallback_indices),
                "stitching_fallback_indices": fallback_indices,
                "displacement": displacement_summary(noisy, denoised),
                "elapsed_seconds": time.time() - cloud_started,
            }
        )
        print(
            f"[{index}/{len(records)}] {key} "
            f"elapsed={entries[-1]['elapsed_seconds']:.2f}s "
            f"fallback={len(fallback_indices)}",
            flush=True,
        )
        jt.gc()

    manifest = {
        "experiment": "PLR-Jittor-native-3DMambaIPF-inference",
        "framework": "Jittor",
        "pytorch_runtime": False,
        "count": len(entries),
        "dtype": "float32",
        "elapsed_seconds": time.time() - started,
        "checkpoint": str(checkpoint),
        "checkpoint_metadata": checkpoint_metadata,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "patch_size": args.patch_size,
        "seed_k": args.seed_k,
        "seed_k_alpha": args.seed_k_alpha,
        "num_modules": args.num_modules,
        "frame_knn": args.frame_knn,
        "noise_decay": args.noise_decay,
        "seed": args.seed,
        "merge_strategy": args.merge_strategy,
        "softmax_temperature": args.softmax_temperature,
        "merge_formula": (
            "exp(-(d-d_min)/0.10)"
            if args.merge_strategy == "softmax010"
            else "minimum normalized seed distance"
        ),
        "input_cyclic_shift_requested": args.input_cyclic_shift,
        "entries": entries,
    }
    write_json_new(manifest_path, manifest)
    print(json.dumps({"count": len(entries), "manifest": str(manifest_path)}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PLRError as exc:
        print(f"ERROR: {exc}", flush=True)
        raise SystemExit(2)
