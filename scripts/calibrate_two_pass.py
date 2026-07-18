#!/usr/bin/env python3
"""Fuse a second denoising pass with a confidence-gated correction.

The first-pass adaptive prediction remains the anchor.  A second model pass is
used as a self-consistency probe: ordinary clouds receive a positive residual
correction, while clouds whose mean/var alpha was clipped to its lower bound
move away from the second-pass direction to avoid further over-smoothing.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict

import numpy as np


class TwoPassError(RuntimeError):
    pass


def discover(root: Path) -> Dict[str, Path]:
    return {
        path.parent.relative_to(root).as_posix(): path
        for path in sorted(root.glob("**/denoised.npy"))
    }


def align_predictions(
    name: str,
    reference_keys,
    values: Dict[str, Path],
) -> Dict[str, Path]:
    aligned = {}
    used = set()
    for key in reference_keys:
        matches = [
            (candidate, path)
            for candidate, path in values.items()
            if candidate == key
            or candidate.endswith("/" + key)
            or key.endswith("/" + candidate)
        ]
        if len(matches) != 1:
            raise TwoPassError(
                f"{name} has {len(matches)} matches for {key}"
            )
        candidate, path = matches[0]
        aligned[key] = path
        used.add(candidate)
    extras = sorted(set(values).difference(used))
    if extras:
        raise TwoPassError(f"{name} has unmatched prediction {extras[0]}")
    return aligned


def load_alphas(path: Path) -> Dict[str, float]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames is None or not {"key", "alpha"}.issubset(reader.fieldnames):
            raise TwoPassError("alpha manifest must contain key and alpha columns")
        result = {row["key"]: float(row["alpha"]) for row in reader}
    if not result:
        raise TwoPassError("alpha manifest is empty")
    return result


def fuse_tree(
    first_dir: Path,
    adaptive_dir: Path,
    second_dir: Path,
    out_dir: Path,
    alpha_manifest: Path,
    expected_count: int,
    lower_alpha: float,
    lower_beta: float,
    regular_beta: float,
    manifest_path: Path,
) -> None:
    if out_dir.exists():
        raise TwoPassError(f"output directory exists: {out_dir}")
    first = discover(first_dir)
    adaptive = discover(adaptive_dir)
    second = discover(second_dir)
    alphas = load_alphas(alpha_manifest)
    keys = set(first)
    if len(keys) != expected_count:
        raise TwoPassError(f"expected {expected_count} first-pass clouds, found {len(keys)}")
    adaptive = align_predictions("adaptive", keys, adaptive)
    second = align_predictions("second", keys, second)
    if set(alphas) != keys:
        missing = sorted(keys.difference(alphas))
        extra = sorted(set(alphas).difference(keys))
        raise TwoPassError(
            f"alpha keys do not match predictions; missing={missing[:1]} extra={extra[:1]}"
        )

    rows = ["key\talpha\tbeta\tsecond_displacement_mean"]
    try:
        for index, key in enumerate(sorted(keys), 1):
            first_points = np.load(first[key], allow_pickle=False).astype(np.float64)
            adaptive_points = np.load(adaptive[key], allow_pickle=False).astype(np.float64)
            second_points = np.load(second[key], allow_pickle=False).astype(np.float64)
            if first_points.shape != adaptive_points.shape or first_points.shape != second_points.shape:
                raise TwoPassError(f"shape mismatch for {key}")
            if not (
                np.isfinite(first_points).all()
                and np.isfinite(adaptive_points).all()
                and np.isfinite(second_points).all()
            ):
                raise TwoPassError(f"non-finite coordinates for {key}")
            alpha = alphas[key]
            beta = lower_beta if alpha <= lower_alpha + 1e-8 else regular_beta
            correction = second_points - first_points
            output = adaptive_points + beta * correction
            target = out_dir / key / "denoised.npy"
            target.parent.mkdir(parents=True, exist_ok=True)
            np.save(target, output.astype(np.float32))
            rows.append(
                f"{key}\t{alpha:.8f}\t{beta:.8f}\t"
                f"{np.linalg.norm(correction, axis=1).mean():.10g}"
            )
            if index % 25 == 0 or index == len(keys):
                print(f"fused: {index}/{len(keys)}", flush=True)
    except Exception:
        marker = out_dir / ".incomplete"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("two-pass fusion failed\n", encoding="utf-8")
        raise

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-dir", type=Path, required=True)
    parser.add_argument("--adaptive-dir", type=Path, required=True)
    parser.add_argument("--second-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--alpha-manifest", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--lower-alpha", type=float, default=0.97)
    parser.add_argument("--lower-beta", type=float, default=-0.30)
    parser.add_argument("--regular-beta", type=float, default=0.45)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fuse_tree(
        args.first_dir.resolve(),
        args.adaptive_dir.resolve(),
        args.second_dir.resolve(),
        args.out_dir.resolve(),
        args.alpha_manifest.resolve(),
        args.expected_count,
        args.lower_alpha,
        args.lower_beta,
        args.regular_beta,
        args.manifest.resolve(),
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TwoPassError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
