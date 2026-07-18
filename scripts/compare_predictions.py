#!/usr/bin/env python3
"""Compare several prediction trees with paired per-cloud bootstrap intervals."""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evaluate  # noqa: E402


class ComparisonError(RuntimeError):
    pass


def _parse_prediction(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("prediction must use LABEL=PATH")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("prediction label must not be empty")
    return label, Path(raw_path).expanduser().resolve()


def _discover(root: Path, filename: str) -> Dict[str, Path]:
    return {
        path.parent.relative_to(root).as_posix(): path
        for path in sorted(root.glob(f"**/{filename}"))
    }


def _mesh_paths(root: Path, filename: str) -> Dict[str, Path]:
    suffix = Path(filename)
    result = {}
    for path in sorted(root.glob(f"**/{filename}")):
        model_dir = path
        for _ in suffix.parts:
            model_dir = model_dir.parent
        result[model_dir.relative_to(root).as_posix()] = path
    return result


def _align_predictions(
    label: str,
    mapping: Mapping[str, Path],
    keys: Sequence[str],
) -> Dict[str, Path]:
    """Align writer trees that may retain an input-root prefix (e.g. localtest2/)."""
    aligned = {}
    used = set()
    for key in keys:
        matches = [
            (candidate_key, path)
            for candidate_key, path in mapping.items()
            if candidate_key == key or candidate_key.endswith("/" + key)
        ]
        if len(matches) != 1:
            raise ComparisonError(
                f"prediction tree {label} has {len(matches)} matches for {key}"
            )
        candidate_key, path = matches[0]
        aligned[key] = path
        used.add(candidate_key)
    extras = sorted(set(mapping).difference(used))
    if extras:
        raise ComparisonError(
            f"prediction tree {label} has an unmatched prediction: {extras[0]}"
        )
    return aligned


def _evaluate_cloud(task):
    key, clean_path, noisy_path, mesh_path, predictions = task
    clean = evaluate.load_pointcloud(str(clean_path))
    noisy = evaluate.load_pointcloud(str(noisy_path))
    mesh_v, mesh_f = evaluate.load_mesh_vf(str(mesh_path))
    if mesh_v is None or mesh_f is None:
        raise ComparisonError(f"failed to load mesh for {key}: {mesh_path}")

    cd_noisy = evaluate.chamfer_distance(noisy, clean, normalize=True)
    p2s_noisy = evaluate.point_to_surface_distance(
        noisy, mesh_v, mesh_f, normalize_ref_pc=clean
    )
    rows = []
    for label, prediction_path in predictions:
        prediction = evaluate.load_pointcloud(str(prediction_path))
        cd_pred = evaluate.chamfer_distance(prediction, clean, normalize=True)
        p2s_pred = evaluate.point_to_surface_distance(
            prediction, mesh_v, mesh_f, normalize_ref_pc=clean
        )
        cd_score = evaluate.metric_to_score(cd_pred, cd_noisy)
        p2s_score = evaluate.metric_to_score(p2s_pred, p2s_noisy)
        rows.append(
            (
                label,
                key,
                cd_pred,
                cd_noisy,
                cd_score,
                p2s_pred,
                p2s_noisy,
                p2s_score,
                0.5 * (cd_score + p2s_score),
            )
        )
    return rows


def _bootstrap_ci(
    differences: np.ndarray,
    samples: int,
    seed: int,
) -> Tuple[float, float]:
    if differences.ndim != 1 or differences.size == 0:
        raise ComparisonError("bootstrap differences must be a non-empty vector")
    if samples <= 0:
        raise ComparisonError("--bootstrap-samples must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, differences.size, size=(samples, differences.size)
    )
    means = differences[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _write_rows(path: Path, rows: Iterable[Sequence[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "label\tkey\tcd_pred\tcd_noisy\tcd_score\t"
        "p2s_pred\tp2s_noisy\tp2s_score\ttotal_score"
    )
    lines = [header]
    for row in rows:
        label, key, *numbers = row
        lines.append(
            "\t".join(
                [str(label), str(key)]
                + [f"{float(value):.12g}" for value in numbers]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        type=_parse_prediction,
        help="repeat LABEL=PATH for every prediction tree",
    )
    parser.add_argument("--reference", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--pred-filename", default="denoised.npy")
    parser.add_argument("--clean-filename", default="clean.npy")
    parser.add_argument("--noisy-filename", default="noisy.npy")
    parser.add_argument("--mesh-filename", default="models/model_normalized.obj")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--output-tsv", type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    labels = [label for label, _ in args.prediction]
    if len(set(labels)) != len(labels):
        raise ComparisonError("prediction labels must be unique")
    if args.reference not in labels:
        raise ComparisonError(f"unknown reference label: {args.reference}")

    data_dir = args.data_dir.expanduser().resolve()
    clean = _discover(data_dir, args.clean_filename)
    noisy = _discover(data_dir, args.noisy_filename)
    meshes = _mesh_paths(data_dir, args.mesh_filename)
    discovered_predictions: Mapping[str, Dict[str, Path]] = {
        label: _discover(path, args.pred_filename)
        for label, path in args.prediction
    }
    keys = sorted(set(clean) & set(noisy) & set(meshes))
    if not keys:
        raise ComparisonError(f"no complete samples found below {data_dir}")
    prediction_maps: Mapping[str, Dict[str, Path]] = {
        label: _align_predictions(label, mapping, keys)
        for label, mapping in discovered_predictions.items()
    }

    tasks = [
        (
            key,
            clean[key],
            noisy[key],
            meshes[key],
            [(label, prediction_maps[label][key]) for label in labels],
        )
        for key in keys
    ]
    workers = args.workers if args.workers > 0 else min(cpu_count(), 16)
    if workers > 1:
        with Pool(workers) as pool:
            nested_rows = pool.map(_evaluate_cloud, tasks)
    else:
        nested_rows = [_evaluate_cloud(task) for task in tasks]
    rows = [row for cloud_rows in nested_rows for row in cloud_rows]
    if args.output_tsv is not None:
        _write_rows(args.output_tsv.expanduser().resolve(), rows)

    by_label: Dict[str, List[Tuple[object, ...]]] = {
        label: [] for label in labels
    }
    for row in rows:
        by_label[str(row[0])].append(row)
    reference = np.asarray(
        [float(row[-1]) for row in by_label[args.reference]], dtype=np.float64
    )

    print("label\tCD\tP2S\ttotal\tdelta\t95% CI")
    for index, label in enumerate(labels):
        candidate_rows = by_label[label]
        cd_score = float(np.mean([float(row[4]) for row in candidate_rows]))
        p2s_score = float(np.mean([float(row[7]) for row in candidate_rows]))
        total = float(np.mean([float(row[8]) for row in candidate_rows]))
        candidate = np.asarray(
            [float(row[-1]) for row in candidate_rows], dtype=np.float64
        )
        differences = candidate - reference
        low, high = _bootstrap_ci(
            differences, args.bootstrap_samples, args.seed + index
        )
        print(
            f"{label}\t{cd_score:.8f}\t{p2s_score:.8f}\t{total:.8f}\t"
            f"{differences.mean():+.8f}\t[{low:+.8f},{high:+.8f}]"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ComparisonError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
