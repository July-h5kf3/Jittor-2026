#!/usr/bin/env python3
"""Generate the frozen Jittor StraightPCF adaptive two-pass ROT reference."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import jittor as jt
import numpy as np

from plr3d.io import (
    PLRError,
    load_cloud,
    read_keys,
    save_cloud_new,
    sha256_file,
    write_json_new,
)

PACKAGE_ROOT = Path(__file__).resolve().parent
ROT_ROOT = PACKAGE_ROOT / "rot_jittor"
if str(ROT_ROOT) not in sys.path:
    sys.path.insert(0, str(ROT_ROOT))

from src.model.straightpcf import StraightPCFModule  # noqa: E402


MODEL_CONFIG = {
    "stage": "spcf",
    "frame_knn": 32,
    "num_train_points": 128,
    "feat_embedding_dim": 256,
    "decoder_hidden_dim": 64,
    "dsm_sigma": 0.01,
    "num_modules": 4,
    "tot_its": 2,
    "decoder_type": "graph",
    "distance_estimation": True,
    "predict_alpha": 1.05,
    "predict_passes": 1,
    "predict_tta": 0,
    "predict_fusion": False,
    "film": True,
    "multiscale": False,
    "distance_multiscale": True,
}
BASE_ALPHA = 1.05
ALPHA_MIN = 0.97
ALPHA_MAX = 1.10
PROFILE_INTERCEPT = 2.1392951014215233
PROFILE_COEFFICIENTS = np.asarray(
    [0.17281014800429867, 0.025992874831480398], dtype=np.float64
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--key-list", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=200)
    parser.add_argument("--expected-points", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args(argv)


def load_model(checkpoint: Path) -> StraightPCFModule:
    model = StraightPCFModule(dict(MODEL_CONFIG), {})
    state = jt.load(str(checkpoint))
    if not isinstance(state, dict):
        raise PLRError("StraightPCF checkpoint must be a state mapping")
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    shape_mismatch = [
        (key, tuple(expected[key].shape), tuple(state[key].shape))
        for key in sorted(set(expected) & set(state))
        if tuple(expected[key].shape) != tuple(state[key].shape)
    ]
    if missing or unexpected or shape_mismatch:
        raise PLRError(
            f"ROT state mismatch: missing={missing[:3]} "
            f"unexpected={unexpected[:3]} shape={shape_mismatch[:3]}"
        )
    model.load_parameters(state)
    model.eval()
    model.set_predict(True)
    return model


def predict_cloud(model: StraightPCFModule, cloud: np.ndarray) -> np.ndarray:
    # The historical NPY loader promoted inputs to float64 before Jittor collation.
    batch = np.ascontiguousarray(cloud.astype(np.float64))[None, ...]
    record = model.predict_step({"pc_noisy": batch})[0]
    result = np.ascontiguousarray(record["pc_denoised"].astype(np.float32))
    if result.shape != cloud.shape or not np.isfinite(result).all():
        raise PLRError(f"invalid ROT prediction: {result.shape} {result.dtype}")
    return result


def adaptive_alpha(noisy: np.ndarray, prediction: np.ndarray) -> Tuple[np.ndarray, float, float, float, float]:
    displacement = np.linalg.norm(
        prediction.astype(np.float64) - noisy.astype(np.float64), axis=1
    )
    mean = float(displacement.mean())
    variance = float(displacement.var())
    features = np.asarray(
        [np.log(mean + 1e-12), np.log(variance + 1e-16)], dtype=np.float64
    )
    raw_alpha = float(PROFILE_INTERCEPT + np.dot(PROFILE_COEFFICIENTS, features))
    alpha = float(np.clip(raw_alpha, ALPHA_MIN, ALPHA_MAX))
    output = noisy.astype(np.float64) + (alpha / BASE_ALPHA) * (
        prediction.astype(np.float64) - noisy.astype(np.float64)
    )
    return np.ascontiguousarray(output.astype(np.float32)), alpha, raw_alpha, mean, variance


def robust_scores(values: Dict[str, float]) -> Dict[str, float]:
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


def save_tsv(path: Path, header, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    jt.flags.use_cuda = 1 if args.device == "cuda" else 0
    random.seed(args.seed)
    np.random.seed(args.seed)
    jt.set_global_seed(args.seed)

    input_root = args.input_root.expanduser().resolve()
    key_list = args.key_list.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    for path in (input_root, key_list, checkpoint):
        if not path.exists():
            raise PLRError(f"missing ROT input: {path}")
    if output_root.exists():
        raise PLRError(f"ROT output already exists: {output_root}")
    keys = read_keys(key_list)
    if len(keys) != args.expected_count:
        raise PLRError(f"expected {args.expected_count} keys, found {len(keys)}")

    raw_root = output_root / "raw" / "runtime_input"
    adaptive_root = output_root / "adaptive" / "runtime_input"
    second_root = output_root / "pass2" / "runtime_input"
    fused_root = output_root / "fused" / "runtime_input"
    canonical_root = output_root / "canonical"
    evidence_root = output_root / "evidence"
    model = load_model(checkpoint)

    arrays = {}
    alpha_rows = []
    correction_statistics = {}
    try:
        for index, key in enumerate(keys, 1):
            noisy_path = input_root.joinpath(*key.split("/"), "noisy.npy")
            noisy = load_cloud(noisy_path, expected_points=args.expected_points)
            first = predict_cloud(model, noisy)
            adaptive, alpha, raw_alpha, mean, variance = adaptive_alpha(noisy, first)
            second = predict_cloud(model, adaptive)

            save_cloud_new(raw_root.joinpath(*key.split("/"), "denoised.npy"), first)
            save_cloud_new(
                adaptive_root.joinpath(*key.split("/"), "denoised.npy"), adaptive
            )
            save_cloud_new(
                second_root.joinpath(*key.split("/"), "denoised.npy"), second
            )
            correction = second.astype(np.float64) - first.astype(np.float64)
            norms = np.linalg.norm(correction, axis=1)
            correction_mean = float(norms.mean())
            correction_variance = float(norms.var())
            correction_cv = float(norms.std() / (correction_mean + 1e-15))
            correction_statistics[key] = correction_cv
            arrays[key] = (adaptive, correction, alpha, raw_alpha, correction_mean, correction_variance)
            alpha_rows.append((key, alpha, raw_alpha, mean, variance))
            print(f"ROT passes: {index}/{len(keys)} {key}", flush=True)

        scores = robust_scores(correction_statistics)
        fusion_rows = []
        entries = []
        for index, key in enumerate(keys, 1):
            adaptive, correction, alpha, raw_alpha, corr_mean, corr_var = arrays[key]
            if raw_alpha <= 0.96 + 1e-8:
                beta = -0.30
            else:
                beta = float(np.clip(0.45 * (1.0 + 0.50 * scores[key]), 0.36, 0.54))
            fused = np.ascontiguousarray(
                (adaptive.astype(np.float64) + beta * correction).astype(np.float32)
            )
            fused_path = fused_root.joinpath(*key.split("/"), "denoised.npy")
            canonical_path = canonical_root.joinpath(*key.split("/"), "denoised.npy")
            save_cloud_new(fused_path, fused)
            canonical_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(fused_path, canonical_path)
            except OSError:
                shutil.copy2(fused_path, canonical_path)
            entries.append({"key": key, "sha256": sha256_file(canonical_path)})
            fusion_rows.append(
                (key, alpha, raw_alpha, "cv", scores[key], beta, corr_mean, corr_var)
            )
            print(f"ROT fusion: {index}/{len(keys)} {key}", flush=True)

        save_tsv(
            evidence_root / "adaptive.tsv",
            ["key", "alpha", "raw_alpha", "displacement_mean", "displacement_variance"],
            alpha_rows,
        )
        save_tsv(
            evidence_root / "fusion.tsv",
            [
                "key",
                "alpha",
                "gate_alpha",
                "residual_feature",
                "residual_score",
                "beta",
                "second_displacement_mean",
                "second_displacement_variance",
            ],
            fusion_rows,
        )
        write_json_new(
            output_root / "canonical_manifest.json",
            {
                "experiment": "ROT-Jittor-adaptive-two-pass",
                "framework": "Jittor",
                "pytorch_runtime": False,
                "third_party_runtime": False,
                "count": len(entries),
                "checkpoint_sha256": sha256_file(checkpoint),
                "formula": "mean-var alpha; pass2; CV-adaptive beta gamma=0.50",
                "entries": entries,
            },
        )
    except Exception:
        if output_root.exists():
            shutil.rmtree(output_root)
        raise

    print(json.dumps({"count": len(keys), "canonical": str(canonical_root)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PLRError as exc:
        print(f"ERROR: {exc}", flush=True)
        raise SystemExit(2)
