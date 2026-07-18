#!/usr/bin/env python3
"""Average compatible Jittor model checkpoints with fixed weights."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np


class CheckpointAverageError(RuntimeError):
    pass


def average_state_dicts(
    states: Sequence[Mapping[str, np.ndarray]],
    weights: Sequence[float],
) -> Dict[str, np.ndarray]:
    if len(states) < 2:
        raise CheckpointAverageError("at least two checkpoints are required")
    if len(states) != len(weights):
        raise CheckpointAverageError("weights/checkpoint count mismatch")
    normalized = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(normalized).all() or np.any(normalized < 0):
        raise CheckpointAverageError("weights must be finite and non-negative")
    if normalized.sum() <= 0:
        raise CheckpointAverageError("at least one weight must be positive")
    normalized /= normalized.sum()

    keys = list(states[0])
    for index, state in enumerate(states[1:], 1):
        if set(state) != set(keys):
            raise CheckpointAverageError(
                f"checkpoint {index} has different parameter keys"
            )

    result = {}
    for key in keys:
        arrays = [np.asarray(state[key]) for state in states]
        shape = arrays[0].shape
        if any(array.shape != shape for array in arrays[1:]):
            raise CheckpointAverageError(f"shape mismatch for {key}")
        dtype = arrays[0].dtype
        if np.issubdtype(dtype, np.floating):
            value = sum(
                float(weight) * array.astype(np.float64)
                for weight, array in zip(normalized, arrays)
            )
            result[key] = value.astype(dtype)
        else:
            if any(not np.array_equal(arrays[0], array) for array in arrays[1:]):
                raise CheckpointAverageError(
                    f"non-floating parameter differs for {key}"
                )
            result[key] = arrays[0].copy()
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", type=Path, required=True)
    parser.add_argument("--weights")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    checkpoints = [path.expanduser().resolve() for path in args.checkpoint]
    if args.output.exists():
        raise CheckpointAverageError(f"output already exists: {args.output}")
    if args.weights is None:
        weights = [1.0] * len(checkpoints)
    else:
        try:
            weights = [float(value) for value in args.weights.split(",")]
        except ValueError as exc:
            raise CheckpointAverageError(
                "--weights must be comma-separated numbers"
            ) from exc

    try:
        import jittor as jt
    except ImportError as exc:
        raise CheckpointAverageError("Jittor is required to load checkpoints") from exc
    loaded = [jt.load(str(path)) for path in checkpoints]
    numpy_states = [
        {key: value.numpy() for key, value in state.items()} for state in loaded
    ]
    averaged = average_state_dicts(numpy_states, weights)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    jt.save({key: jt.array(value) for key, value in averaged.items()}, str(args.output))
    print(
        f"averaged={len(checkpoints)} parameters={len(averaged)} "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CheckpointAverageError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
