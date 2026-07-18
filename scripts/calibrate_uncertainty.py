#!/usr/bin/env python3
"""Calibrate an aligned prediction ensemble using checkpoint disagreement.

The method uses only noisy inputs and model predictions.  A cloud-level
mean/variance alpha is applied to the ensemble mean, while point-wise
checkpoint disagreement redistributes (but approximately preserves) the
average step length.  Positive gamma is conservative on uncertain points;
negative gamma tests the opposite hypothesis.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.calibrate_predictions import (  # noqa: E402
    BASE_ALPHA,
    PROFILE_NAME,
    estimate_alpha,
    extract_features,
)


class UncertaintyCalibrationError(RuntimeError):
    pass


def _discover(root: Path, filename: str) -> Dict[str, Path]:
    return {
        path.parent.relative_to(root).as_posix(): path
        for path in sorted(root.glob(f"**/{filename}"))
    }


def calibrate_cloud(
    noisy: np.ndarray,
    predictions: Sequence[np.ndarray],
    gamma: float,
    base_alpha: float = BASE_ALPHA,
) -> tuple:
    if len(predictions) < 2:
        raise UncertaintyCalibrationError("at least two predictions are required")
    stack = np.stack(predictions, axis=0).astype(np.float64)
    if stack.ndim != 3 or stack.shape[2] != 3:
        raise UncertaintyCalibrationError(f"invalid prediction stack: {stack.shape}")
    if noisy.shape != stack.shape[1:]:
        raise UncertaintyCalibrationError(
            f"noisy/prediction shape mismatch: {noisy.shape} vs {stack.shape[1:]}"
        )
    if not np.isfinite(stack).all() or not np.isfinite(noisy).all():
        raise UncertaintyCalibrationError("non-finite input coordinate")
    if base_alpha <= 0:
        raise UncertaintyCalibrationError("base alpha must be positive")

    mean_prediction = stack.mean(axis=0)
    features = extract_features(noisy, mean_prediction, "ensemble")
    alpha = estimate_alpha(features, PROFILE_NAME)
    disagreement = np.sqrt(
        np.mean(np.sum((stack - mean_prediction[None, ...]) ** 2, axis=2), axis=0)
    )
    scale = float(np.quantile(disagreement, 0.90))
    if scale <= 1e-15:
        normalized = np.zeros_like(disagreement)
    else:
        normalized = np.clip(disagreement / scale, 0.0, 1.0)
    centered = normalized - normalized.mean()
    factors = np.clip(1.0 - gamma * centered, 0.90, 1.10)
    output = noisy + (
        (alpha / base_alpha)
        * factors[:, None]
        * (mean_prediction - noisy)
    )
    return (
        output.astype(np.float32),
        alpha,
        disagreement,
        factors,
    )


def calibrate_trees(
    input_dirs: Sequence[Path],
    noisy_dir: Path,
    output_dir: Path,
    gamma: float,
    expected_count: int,
    manifest_path: Path,
) -> None:
    if output_dir.exists():
        raise UncertaintyCalibrationError(
            f"output directory already exists: {output_dir}"
        )
    maps = [_discover(path, "denoised.npy") for path in input_dirs]
    if len(maps) < 2:
        raise UncertaintyCalibrationError("at least two input trees are required")
    keys = sorted(maps[0])
    if len(keys) != expected_count:
        raise UncertaintyCalibrationError(
            f"expected {expected_count} predictions, found {len(keys)}"
        )
    for index, mapping in enumerate(maps[1:], 1):
        if set(mapping) != set(keys):
            raise UncertaintyCalibrationError(
                f"prediction tree {index} does not match the first tree"
            )
    noisy_map = _discover(noisy_dir, "noisy.npy")
    missing = sorted(set(keys).difference(noisy_map))
    if missing:
        raise UncertaintyCalibrationError(f"missing noisy cloud for {missing[0]}")

    rows = [
        "key\talpha\tdisagreement_mean\tdisagreement_p90\t"
        "factor_min\tfactor_max"
    ]
    try:
        for index, key in enumerate(keys, 1):
            noisy = np.load(noisy_map[key], allow_pickle=False).astype(np.float64)
            predictions = [
                np.load(mapping[key], allow_pickle=False).astype(np.float64)
                for mapping in maps
            ]
            output, alpha, disagreement, factors = calibrate_cloud(
                noisy, predictions, gamma
            )
            target = output_dir / key / "denoised.npy"
            target.parent.mkdir(parents=True, exist_ok=True)
            np.save(target, output)
            rows.append(
                f"{key}\t{alpha:.8f}\t{disagreement.mean():.10g}\t"
                f"{np.quantile(disagreement, 0.90):.10g}\t"
                f"{factors.min():.8f}\t{factors.max():.8f}"
            )
            if index % 25 == 0 or index == len(keys):
                print(f"calibrated: {index}/{len(keys)}", flush=True)
    except Exception:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        raise

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", action="append", type=Path, required=True)
    parser.add_argument("--noisy-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    calibrate_trees(
        [path.expanduser().resolve() for path in args.input_dir],
        args.noisy_dir.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        args.gamma,
        args.expected_count,
        args.manifest.expanduser().resolve(),
    )
    print(f"gamma={args.gamma:g} output={args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except UncertaintyCalibrationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
