#!/usr/bin/env python3
"""Adapt the two-pass residual weight from cloud-level residual statistics.

This is a label-free inference rule inspired by adaptive diffusion schedules.
It can use either the mean correction magnitude or its coefficient of
variation (standard deviation divided by mean).  The latter separates a
spatially uneven, uncertain correction from a uniformly scaled correction and
is invariant to the absolute noise scale.  New calibration manifests retain
the unclipped alpha so the lower-alpha gate can distinguish a marginal clip
from a genuinely out-of-distribution low-alpha cloud.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.calibrate_two_pass import (  # noqa: E402
    TwoPassError,
    align_predictions,
    discover,
    load_alphas,
)


def _robust_scores(values: Dict[str, float]) -> Dict[str, float]:
    keys = sorted(values)
    vector = np.asarray([np.log(values[key] + 1e-15) for key in keys])
    median = float(np.median(vector))
    mad = float(np.median(np.abs(vector - median)))
    if mad <= 1e-15:
        normalized = np.zeros_like(vector)
    else:
        normalized = (vector - median) / (1.4826 * mad)
    normalized = np.clip(normalized, -1.0, 1.0)
    return {key: float(value) for key, value in zip(keys, normalized)}


def _correction_statistic(norms: np.ndarray, feature: str) -> float:
    if norms.ndim != 1 or norms.size == 0:
        raise TwoPassError("correction norms must be a non-empty vector")
    mean = float(norms.mean())
    if feature == "mean":
        return mean
    if feature == "cv":
        return float(norms.std() / (mean + 1e-15))
    raise TwoPassError(f"unknown residual feature: {feature}")


def _load_gate_alphas(
    path: Path,
    clipped_alphas: Dict[str, float],
) -> Tuple[Dict[str, float], bool]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        has_raw_alpha = bool(reader.fieldnames and "raw_alpha" in reader.fieldnames)
        if not has_raw_alpha:
            return dict(clipped_alphas), False
        raw_alphas = {row["key"]: float(row["raw_alpha"]) for row in reader}
    if set(raw_alphas) != set(clipped_alphas):
        raise TwoPassError("raw alpha keys do not match clipped alpha keys")
    return raw_alphas, True


def fuse_tree(
    first_dir: Path,
    adaptive_dir: Path,
    second_dir: Path,
    out_dir: Path,
    alpha_manifest: Path,
    expected_count: int,
    lower_alpha: float,
    raw_lower_alpha: float,
    lower_beta: float,
    regular_beta: float,
    gamma: float,
    score_feature: str,
    beta_min: float,
    beta_max: float,
    manifest_path: Path,
) -> None:
    if out_dir.exists():
        raise TwoPassError(f"output directory exists: {out_dir}")
    first = discover(first_dir)
    keys = set(first)
    if len(keys) != expected_count:
        raise TwoPassError(
            f"expected {expected_count} first-pass clouds, found {len(keys)}"
        )
    adaptive = align_predictions("adaptive", keys, discover(adaptive_dir))
    second = align_predictions("second", keys, discover(second_dir))
    alphas = load_alphas(alpha_manifest)
    gate_alphas, uses_raw_alpha = _load_gate_alphas(alpha_manifest, alphas)
    if set(alphas) != keys:
        raise TwoPassError("alpha keys do not match predictions")
    if not -1 <= gamma <= 1:
        raise TwoPassError("gamma must be in [-1, 1]")
    if beta_min > beta_max:
        raise TwoPassError("beta_min must not exceed beta_max")
    if score_feature not in {"mean", "cv"}:
        raise TwoPassError(f"unknown residual feature: {score_feature}")

    correction_means = {}
    correction_variances = {}
    correction_statistics = {}
    arrays = {}
    for key in sorted(keys):
        first_points = np.load(first[key], allow_pickle=False).astype(np.float64)
        adaptive_points = np.load(adaptive[key], allow_pickle=False).astype(np.float64)
        second_points = np.load(second[key], allow_pickle=False).astype(np.float64)
        if (
            first_points.shape != adaptive_points.shape
            or first_points.shape != second_points.shape
        ):
            raise TwoPassError(f"shape mismatch for {key}")
        correction = second_points - first_points
        correction_norms = np.linalg.norm(correction, axis=1)
        correction_means[key] = float(correction_norms.mean())
        correction_variances[key] = float(correction_norms.var())
        correction_statistics[key] = _correction_statistic(
            correction_norms,
            score_feature,
        )
        arrays[key] = (adaptive_points, correction)

    scores = _robust_scores(correction_statistics)
    rows = [
        "key\talpha\tgate_alpha\tresidual_feature\tresidual_score\tbeta\t"
        "second_displacement_mean\tsecond_displacement_variance"
    ]
    try:
        for index, key in enumerate(sorted(keys), 1):
            alpha = alphas[key]
            gate_alpha = gate_alphas[key]
            gate_threshold = raw_lower_alpha if uses_raw_alpha else lower_alpha
            if gate_alpha <= gate_threshold + 1e-8:
                beta = lower_beta
            else:
                beta = regular_beta * (1.0 + gamma * scores[key])
                beta = float(np.clip(beta, beta_min, beta_max))
            adaptive_points, correction = arrays[key]
            output = adaptive_points + beta * correction
            target = out_dir / key / "denoised.npy"
            target.parent.mkdir(parents=True, exist_ok=True)
            np.save(target, output.astype(np.float32))
            rows.append(
                f"{key}\t{alpha:.8f}\t{gate_alpha:.8f}\t{score_feature}\t"
                f"{scores[key]:.8f}\t"
                f"{beta:.8f}\t{correction_means[key]:.10g}\t"
                f"{correction_variances[key]:.10g}"
            )
            if index % 25 == 0 or index == len(keys):
                print(f"fused: {index}/{len(keys)}", flush=True)
    except Exception:
        marker = out_dir / ".incomplete"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("adaptive two-pass fusion failed\n", encoding="utf-8")
        raise

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-dir", type=Path, required=True)
    parser.add_argument("--adaptive-dir", type=Path, required=True)
    parser.add_argument("--second-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--alpha-manifest", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--lower-alpha", type=float, default=0.97)
    parser.add_argument("--raw-lower-alpha", type=float, default=0.96)
    parser.add_argument("--lower-beta", type=float, default=-0.30)
    parser.add_argument("--regular-beta", type=float, default=0.45)
    parser.add_argument("--gamma", type=float, default=0.20)
    parser.add_argument(
        "--score-feature",
        choices=("mean", "cv"),
        default="mean",
        help="cloud statistic used to adapt beta; cv is std/mean",
    )
    parser.add_argument("--beta-min", type=float, default=0.36)
    parser.add_argument("--beta-max", type=float, default=0.54)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    fuse_tree(
        args.first_dir.resolve(),
        args.adaptive_dir.resolve(),
        args.second_dir.resolve(),
        args.out_dir.resolve(),
        args.alpha_manifest.resolve(),
        args.expected_count,
        args.lower_alpha,
        args.raw_lower_alpha,
        args.lower_beta,
        args.regular_beta,
        args.gamma,
        args.score_feature,
        args.beta_min,
        args.beta_max,
        args.manifest.resolve(),
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TwoPassError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
