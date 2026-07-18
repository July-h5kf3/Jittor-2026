#!/usr/bin/env python3
"""Average point-aligned prediction trees with fixed, normalized weights."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


class EnsembleError(RuntimeError):
    pass


def _discover(root: Path, filename: str) -> Dict[str, Path]:
    return {
        path.parent.relative_to(root).as_posix(): path
        for path in sorted(root.glob(f"**/{filename}"))
    }


def _align_map(
    reference_keys: Sequence[str],
    mapping: Dict[str, Path],
) -> Dict[str, Path]:
    aligned = {}
    used = set()
    for key in reference_keys:
        matches = [
            (candidate, path)
            for candidate, path in mapping.items()
            if candidate == key
            or candidate.endswith("/" + key)
            or key.endswith("/" + candidate)
        ]
        if len(matches) != 1:
            raise EnsembleError(
                f"prediction tree has {len(matches)} matches for {key}"
            )
        candidate, path = matches[0]
        aligned[key] = path
        used.add(candidate)
    extras = sorted(set(mapping).difference(used))
    if extras:
        raise EnsembleError(f"unmatched prediction: {extras[0]}")
    return aligned


def ensemble_trees(
    input_dirs: Sequence[Path],
    weights: Sequence[float],
    output_dir: Path,
    filename: str,
    expected_count: int,
) -> None:
    if len(input_dirs) < 2:
        raise EnsembleError("at least two input trees are required")
    if len(input_dirs) != len(weights):
        raise EnsembleError("the number of weights must match the input trees")
    normalized = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(normalized).all() or np.any(normalized < 0):
        raise EnsembleError("weights must be finite and non-negative")
    total = float(normalized.sum())
    if total <= 0:
        raise EnsembleError("at least one weight must be positive")
    normalized /= total
    if output_dir.exists():
        raise EnsembleError(f"output directory already exists: {output_dir}")

    maps = [_discover(path, filename) for path in input_dirs]
    keys = sorted(maps[0])
    if len(keys) != expected_count:
        raise EnsembleError(
            f"expected {expected_count} predictions, found {len(keys)}"
        )
    maps = [maps[0]] + [_align_map(keys, mapping) for mapping in maps[1:]]

    try:
        for index, key in enumerate(keys, 1):
            output = None
            shape = None
            for weight, mapping in zip(normalized, maps):
                points = np.load(mapping[key], allow_pickle=False).astype(np.float64)
                if points.ndim != 2 or points.shape[1] != 3:
                    raise EnsembleError(f"invalid point shape {points.shape} for {key}")
                if shape is None:
                    shape = points.shape
                elif points.shape != shape:
                    raise EnsembleError(f"shape mismatch for {key}")
                if not np.isfinite(points).all():
                    raise EnsembleError(f"non-finite coordinate for {key}")
                contribution = float(weight) * points
                output = contribution if output is None else output + contribution
            target = output_dir / key / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            np.save(target, output.astype(np.float32))
            if index % 25 == 0 or index == len(keys):
                print(f"ensembled: {index}/{len(keys)}", flush=True)
    except Exception:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        raise


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", action="append", type=Path, required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--filename", default="denoised.npy")
    parser.add_argument("--expected-count", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        weights: List[float] = [
            float(value.strip()) for value in args.weights.split(",")
        ]
    except ValueError as exc:
        raise EnsembleError("--weights must be comma-separated numbers") from exc
    ensemble_trees(
        [path.expanduser().resolve() for path in args.input_dir],
        weights,
        args.output_dir.expanduser().resolve(),
        args.filename,
        args.expected_count,
    )
    print(
        f"output={args.output_dir} normalized_weights="
        f"{','.join(f'{value / sum(weights):.8f}' for value in weights)}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EnsembleError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
