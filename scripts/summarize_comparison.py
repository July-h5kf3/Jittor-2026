#!/usr/bin/env python3
"""Summarize paired prediction TSVs by metric, category, and held-out category."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path, PurePosixPath
from typing import Dict, Mapping, Optional, Sequence

import numpy as np


METRICS = ("cd_score", "p2s_score", "total_score")


class SummaryError(RuntimeError):
    pass


def _category(key: str) -> str:
    parts = PurePosixPath(key).parts
    try:
        index = parts.index("shapenet")
    except ValueError:
        index = -1
    if index >= 0 and index + 1 < len(parts):
        return parts[index + 1]
    if parts:
        return parts[0]
    raise SummaryError("empty cloud key")


def load_rows(path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"label", "key", *METRICS}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise SummaryError(
                "comparison TSV must contain label, key, cd_score, "
                "p2s_score, and total_score"
            )
        result: Dict[str, Dict[str, Dict[str, float]]] = {}
        for row in reader:
            label = row["label"]
            key = row["key"]
            if key in result.setdefault(label, {}):
                raise SummaryError(f"duplicate row for {label}/{key}")
            values = {metric: float(row[metric]) for metric in METRICS}
            if not np.isfinite(list(values.values())).all():
                raise SummaryError(f"non-finite score for {label}/{key}")
            result[label][key] = values
    if not result:
        raise SummaryError("comparison TSV is empty")
    return result


def _bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> Sequence[float]:
    if values.ndim != 1 or values.size == 0:
        raise SummaryError("bootstrap input must be a non-empty vector")
    if samples <= 0:
        raise SummaryError("--bootstrap-samples must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    low, high = np.quantile(values[indices].mean(axis=1), [0.025, 0.975])
    return [float(low), float(high)]


def _metric_summary(
    reference: np.ndarray,
    candidate: np.ndarray,
    samples: int,
    seed: int,
) -> Dict[str, object]:
    delta = candidate - reference
    return {
        "reference": float(reference.mean()),
        "candidate": float(candidate.mean()),
        "delta": float(delta.mean()),
        "ci95": _bootstrap_ci(delta, samples, seed),
    }


def summarize(
    rows: Mapping[str, Mapping[str, Mapping[str, float]]],
    reference_label: str,
    candidate_labels: Sequence[str],
    bootstrap_samples: int,
    seed: int,
) -> Dict[str, object]:
    if reference_label not in rows:
        raise SummaryError(f"unknown reference label: {reference_label}")
    reference_rows = rows[reference_label]
    keys = sorted(reference_rows)
    if not keys:
        raise SummaryError("reference label has no rows")

    summaries: Dict[str, object] = {}
    for candidate_index, label in enumerate(candidate_labels):
        if label == reference_label:
            raise SummaryError("candidate must differ from reference")
        if label not in rows:
            raise SummaryError(f"unknown candidate label: {label}")
        candidate_rows = rows[label]
        if set(candidate_rows) != set(keys):
            raise SummaryError(f"candidate keys do not match reference: {label}")

        metric_arrays = {}
        paired = {}
        for metric_index, metric in enumerate(METRICS):
            reference = np.asarray(
                [reference_rows[key][metric] for key in keys], dtype=np.float64
            )
            candidate = np.asarray(
                [candidate_rows[key][metric] for key in keys], dtype=np.float64
            )
            metric_arrays[metric] = (reference, candidate)
            paired[metric] = _metric_summary(
                reference,
                candidate,
                bootstrap_samples,
                seed + 100 * candidate_index + metric_index,
            )

        total_delta = metric_arrays["total_score"][1] - metric_arrays["total_score"][0]
        category_keys: Dict[str, list] = {}
        for index, key in enumerate(keys):
            category_keys.setdefault(_category(key), []).append(index)

        categories = {}
        leave_one_out = {}
        all_indices = np.arange(len(keys))
        for category, indices_list in sorted(category_keys.items()):
            indices = np.asarray(indices_list, dtype=np.int64)
            held_in = {}
            held_out = {}
            mask = np.ones(len(keys), dtype=bool)
            mask[indices] = False
            remaining = all_indices[mask]
            for metric in METRICS:
                reference, candidate = metric_arrays[metric]
                held_in[metric] = {
                    "reference": float(reference[indices].mean()),
                    "candidate": float(candidate[indices].mean()),
                    "delta": float((candidate[indices] - reference[indices]).mean()),
                }
                if remaining.size:
                    held_out[metric] = {
                        "delta": float(
                            (candidate[remaining] - reference[remaining]).mean()
                        )
                    }
            categories[category] = {"count": int(indices.size), "metrics": held_in}
            leave_one_out[category] = {
                "remaining_count": int(remaining.size),
                "metrics": held_out,
            }

        loco_total = [
            item["metrics"]["total_score"]["delta"]
            for item in leave_one_out.values()
            if "total_score" in item["metrics"]
        ]
        summaries[label] = {
            "count": len(keys),
            "wins": int((total_delta > 0).sum()),
            "ties": int((total_delta == 0).sum()),
            "metrics": paired,
            "categories": categories,
            "leave_one_category_out": leave_one_out,
            "loco_total_delta_range": (
                [float(min(loco_total)), float(max(loco_total))]
                if loco_total
                else []
            ),
        }

    return {
        "reference": reference_label,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "candidates": summaries,
    }


def _format_ci(values: Sequence[float]) -> str:
    return f"[{values[0]:+.8f},{values[1]:+.8f}]"


def print_summary(summary: Mapping[str, object]) -> None:
    print(
        "candidate\tCD\tP2S\ttotal\tdCD\tCD_CI\tdP2S\tP2S_CI\t"
        "dTotal\ttotal_CI\twins/count\tLOCO_total_range"
    )
    candidates = summary["candidates"]
    assert isinstance(candidates, Mapping)
    for label, raw_item in candidates.items():
        item = raw_item
        metrics = item["metrics"]
        cd = metrics["cd_score"]
        p2s = metrics["p2s_score"]
        total = metrics["total_score"]
        loco = item["loco_total_delta_range"]
        loco_text = "[]" if not loco else f"[{loco[0]:+.8f},{loco[1]:+.8f}]"
        print(
            f"{label}\t{cd['candidate']:.8f}\t{p2s['candidate']:.8f}\t"
            f"{total['candidate']:.8f}\t{cd['delta']:+.8f}\t"
            f"{_format_ci(cd['ci95'])}\t{p2s['delta']:+.8f}\t"
            f"{_format_ci(p2s['ci95'])}\t{total['delta']:+.8f}\t"
            f"{_format_ci(total['ci95'])}\t{item['wins']}/{item['count']}\t"
            f"{loco_text}"
        )
        print("category\tcount\tdCD\tdP2S\tdTotal")
        for category, category_item in item["categories"].items():
            category_metrics = category_item["metrics"]
            print(
                f"{category}\t{category_item['count']}\t"
                f"{category_metrics['cd_score']['delta']:+.8f}\t"
                f"{category_metrics['p2s_score']['delta']:+.8f}\t"
                f"{category_metrics['total_score']['delta']:+.8f}"
            )
        print("leave_out_category\tremaining\tdCD\tdP2S\tdTotal")
        for category, loco_item in item["leave_one_category_out"].items():
            loco_metrics = loco_item["metrics"]
            print(
                f"{category}\t{loco_item['remaining_count']}\t"
                f"{loco_metrics['cd_score']['delta']:+.8f}\t"
                f"{loco_metrics['p2s_score']['delta']:+.8f}\t"
                f"{loco_metrics['total_score']['delta']:+.8f}"
            )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-tsv", type=Path, required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", action="append")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    rows = load_rows(args.input_tsv.expanduser().resolve())
    candidates = args.candidate or sorted(set(rows).difference({args.reference}))
    summary = summarize(
        rows,
        args.reference,
        candidates,
        args.bootstrap_samples,
        args.seed,
    )
    print_summary(summary)
    if args.output_json is not None:
        output = args.output_json.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SummaryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
