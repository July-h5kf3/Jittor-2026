#!/usr/bin/env python3
"""Apply a per-cloud displacement calibration to denoising predictions.

The calibrator uses only the noisy input and the model prediction.  It never
reads clean points or meshes, so the exact same operation can be applied to
the competition test set.  Coefficients were fitted on local2 with a fixed
five-fold protocol; see EXPERIMENTS.md for the corresponding ablation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np


BASE_ALPHA = 1.05
ALPHA_MIN = 0.97
ALPHA_MAX = 1.10

# Raw-space ridge coefficients fitted on the deterministic V4 control.
PROFILE_NAME = "mean-var"
PROFILE_INTERCEPT = 2.1392951014215233
PROFILE_COEFFICIENTS = np.asarray(
    [0.17281014800429867, 0.025992874831480398], dtype=np.float64
)


class CalibrationError(RuntimeError):
    pass


def extract_features(
    noisy: np.ndarray,
    prediction: np.ndarray,
    key: str,
) -> np.ndarray:
    if noisy.shape != prediction.shape or noisy.ndim != 2 or noisy.shape[1] != 3:
        raise CalibrationError(
            f"shape mismatch for {key}: noisy={noisy.shape}, prediction={prediction.shape}"
        )
    if noisy.shape[0] < 3:
        raise CalibrationError(f"point cloud {key} has fewer than three points")
    if not np.isfinite(noisy).all() or not np.isfinite(prediction).all():
        raise CalibrationError(f"non-finite coordinate in {key}")

    displacement = np.linalg.norm(prediction - noisy, axis=1)
    mean = float(displacement.mean())
    variance = float(displacement.var())
    return np.asarray(
        [np.log(mean + 1e-12), np.log(variance + 1e-16)],
        dtype=np.float64,
    )


def estimate_raw_alpha(features: np.ndarray, profile: str) -> float:
    if profile != PROFILE_NAME:
        raise CalibrationError(f"unknown profile: {profile}")
    if features.shape != PROFILE_COEFFICIENTS.shape:
        raise CalibrationError(
            f"profile {profile} expects {len(PROFILE_COEFFICIENTS)} features, "
            f"got {len(features)}"
        )
    return float(PROFILE_INTERCEPT + np.dot(PROFILE_COEFFICIENTS, features))


def estimate_alpha(features: np.ndarray, profile: str) -> float:
    value = estimate_raw_alpha(features, profile)
    return float(np.clip(value, ALPHA_MIN, ALPHA_MAX))


def calibrate_cloud(
    noisy: np.ndarray,
    prediction: np.ndarray,
    alpha: float,
    base_alpha: float = BASE_ALPHA,
) -> np.ndarray:
    if base_alpha <= 0:
        raise CalibrationError("base alpha must be positive")
    output = noisy + (alpha / base_alpha) * (prediction - noisy)
    return output.astype(np.float32)


def _discover(root: Path, filename: str) -> Dict[str, Path]:
    return {
        path.parent.relative_to(root).as_posix(): path
        for path in sorted(root.glob(f"**/{filename}"))
    }


def calibrate_tree(
    pred_dir: Path,
    noisy_dir: Path,
    out_dir: Path,
    profile: str,
    base_alpha: float,
    expected_count: int,
    manifest_path: Path,
) -> None:
    if out_dir.exists():
        raise CalibrationError(f"output directory already exists: {out_dir}")
    predictions = _discover(pred_dir, "denoised.npy")
    noisy_clouds = _discover(noisy_dir, "noisy.npy")
    if len(predictions) != expected_count:
        raise CalibrationError(
            f"expected {expected_count} predictions, found {len(predictions)}"
        )
    missing_noisy = sorted(set(predictions).difference(noisy_clouds))
    if missing_noisy:
        raise CalibrationError(f"missing noisy cloud for {missing_noisy[0]}")

    rows = [
        "key\talpha\traw_alpha\tdisplacement_mean\tdisplacement_variance"
    ]
    try:
        for index, key in enumerate(sorted(predictions), 1):
            prediction = np.load(predictions[key], allow_pickle=False).astype(np.float64)
            noisy = np.load(noisy_clouds[key], allow_pickle=False).astype(np.float64)
            features = extract_features(noisy, prediction, key)
            raw_alpha = estimate_raw_alpha(features, profile)
            alpha = float(np.clip(raw_alpha, ALPHA_MIN, ALPHA_MAX))
            calibrated = calibrate_cloud(noisy, prediction, alpha, base_alpha)
            target = out_dir / key / "denoised.npy"
            target.parent.mkdir(parents=True, exist_ok=True)
            np.save(target, calibrated)
            displacement = np.linalg.norm(prediction - noisy, axis=1)
            rows.append(
                f"{key}\t{alpha:.8f}\t{raw_alpha:.8f}\t"
                f"{displacement.mean():.10g}\t"
                f"{displacement.var():.10g}"
            )
            if index % 25 == 0 or index == len(predictions):
                print(f"calibrated: {index}/{len(predictions)}", flush=True)
    except Exception:
        # Do not silently leave a directory that looks like a complete result.
        marker = out_dir / ".incomplete"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("calibration failed\n", encoding="utf-8")
        raise

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--noisy-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--profile", choices=(PROFILE_NAME,), default=PROFILE_NAME
    )
    parser.add_argument("--base-alpha", type=float, default=BASE_ALPHA)
    parser.add_argument("--expected-count", type=int, default=200)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pred_dir = args.pred_dir.expanduser().resolve()
    noisy_dir = args.noisy_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    manifest = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else out_dir.parent / f"{out_dir.name}_adaptive_alpha.tsv"
    )
    calibrate_tree(
        pred_dir,
        noisy_dir,
        out_dir,
        args.profile,
        args.base_alpha,
        args.expected_count,
        manifest,
    )
    print(f"profile={args.profile} output={out_dir} manifest={manifest}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CalibrationError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
