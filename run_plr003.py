#!/usr/bin/env python3
"""End-to-end PLR-003 expert inference and noisy-only assembly."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import jittor as jt

from plr3d.backend import add_device_argument, configure_device
from plr3d.io import PLRError, read_keys
from plr3d.routing import AIRPLANE, SOFA, TABLE, TAIL_CATEGORIES


def write_list_new(path: Path, keys):
    if path.exists():
        raise PLRError(f"refusing to overwrite list: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(keys) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--key-list", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--rot-root", type=Path, required=True)
    parser.add_argument("--full-dcd-checkpoint", type=Path, required=True)
    parser.add_argument("--airplane-checkpoint", type=Path, required=True)
    parser.add_argument("--table-checkpoint", type=Path, required=True)
    parser.add_argument("--sofa-checkpoint", type=Path, required=True)
    parser.add_argument("--tail-checkpoint", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=200)
    parser.add_argument("--expected-points", type=int, default=50000)
    parser.add_argument("--patch-size", type=int, default=2000)
    parser.add_argument("--seed-k", type=int, default=6)
    parser.add_argument("--seed-k-alpha", type=int, default=20)
    parser.add_argument("--num-modules", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2020)
    add_device_argument(parser)
    args = parser.parse_args()
    configure_device(args.device, jt)

    if args.work_root.exists() or args.output_root.exists() or args.manifest.exists():
        raise PLRError("work/output/manifest target already exists")
    keys = read_keys(args.key_list)
    if len(keys) != args.expected_count:
        raise PLRError(f"expected {args.expected_count} keys, found {len(keys)}")
    groups = {
        "full_dcd": [key for key in keys if key.split("/")[1] == SOFA],
        "airplane": [key for key in keys if key.split("/")[1] == AIRPLANE],
        "table": [key for key in keys if key.split("/")[1] == TABLE],
        "sofa": [key for key in keys if key.split("/")[1] == SOFA],
        "tail": [key for key in keys if key.split("/")[1] in TAIL_CATEGORIES],
    }
    empty_groups = [label for label, group in groups.items() if not group]
    if empty_groups:
        raise PLRError(f"key list has empty required expert groups: {empty_groups}")
    checkpoints = {
        "full_dcd": args.full_dcd_checkpoint,
        "airplane": args.airplane_checkpoint,
        "table": args.table_checkpoint,
        "sofa": args.sofa_checkpoint,
        "tail": args.tail_checkpoint,
    }
    args.work_root.mkdir(parents=True)
    package_root = Path(__file__).resolve().parent
    inference_roots = {}
    for label in ("full_dcd", "airplane", "table", "sofa", "tail"):
        list_path = args.work_root / "lists" / f"{label}.txt"
        output_path = args.work_root / "experts" / label
        manifest_path = args.work_root / "manifests" / f"{label}.json"
        write_list_new(list_path, groups[label])
        command = [
            sys.executable,
            str(package_root / "infer.py"),
            "--checkpoint",
            str(checkpoints[label]),
            "--input-root",
            str(args.input_root),
            "--key-list",
            str(list_path),
            "--output-root",
            str(output_path),
            "--manifest",
            str(manifest_path),
            "--expected-count",
            str(len(groups[label])),
            "--expected-points",
            str(args.expected_points),
            "--patch-size",
            str(args.patch_size),
            "--seed-k",
            str(args.seed_k),
            "--seed-k-alpha",
            str(args.seed_k_alpha),
            "--num-modules",
            str(args.num_modules),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
        ]
        subprocess.run(command, check=True)
        inference_roots[label] = output_path

    command = [
        sys.executable,
        str(package_root / "assemble.py"),
        "--key-list",
        str(args.key_list),
        "--noisy-root",
        str(args.input_root),
        "--full-root",
        str(args.full_root),
        "--rot-root",
        str(args.rot_root),
        "--full-dcd-root",
        str(inference_roots["full_dcd"]),
        "--airplane-root",
        str(inference_roots["airplane"]),
        "--table-root",
        str(inference_roots["table"]),
        "--sofa-root",
        str(inference_roots["sofa"]),
        "--tail-root",
        str(inference_roots["tail"]),
        "--output-root",
        str(args.output_root),
        "--manifest",
        str(args.manifest),
        "--expected-count",
        str(args.expected_count),
        "--expected-points",
        str(args.expected_points),
    ]
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PLRError as exc:
        print(f"ERROR: {exc}", flush=True)
        raise SystemExit(2)
