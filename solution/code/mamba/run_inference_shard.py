#!/usr/bin/env python3
"""Select a deterministic key shard and invoke the frozen Jittor infer entry."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


TASK_ROOT = Path(__file__).resolve().parent
DEFAULT_CODE_ROOT = (
    TASK_ROOT.parents[1] / "submission_packages" / "contest2_NKAI_031" / "code"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, default=DEFAULT_CODE_ROOT)
    parser.add_argument(
        "--inference-script",
        type=Path,
        help="defaults to CODE_ROOT/infer.py; may point to an audited Jittor wrapper",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--key-list", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--patch-size", type=int, default=2000)
    parser.add_argument("--seed-k", type=int, default=6)
    parser.add_argument("--seed-k-alpha", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument(
        "--merge-strategy",
        choices=("softmax010", "best_patch"),
        default="softmax010",
    )
    parser.add_argument("--softmax-temperature", type=float, default=0.10)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")
    keys = [
        line.strip().replace("\\", "/").strip("/")
        for line in args.key_list.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("input key list must be non-empty and unique")
    selected = keys[args.shard_index :: args.shard_count]
    if not selected:
        raise ValueError("shard selects no keys")
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    shard_list = args.output_root.parent / (
        args.output_root.name + ".keys-%03d-of-%03d.txt" %
        (args.shard_index, args.shard_count)
    )
    if shard_list.exists():
        existing = shard_list.read_text(encoding="utf-8").splitlines()
        if existing != selected:
            raise ValueError("existing shard list content mismatch")
    else:
        with shard_list.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(selected) + "\n")
    inference_script = (
        args.inference_script.resolve(strict=True)
        if args.inference_script is not None
        else args.code_root.resolve(strict=True) / "infer.py"
    )
    command = [
        sys.executable,
        str(inference_script),
        "--checkpoint", str(args.checkpoint.resolve(strict=True)),
        "--input-root", str(args.input_root.resolve(strict=True)),
        "--output-root", str(args.output_root),
        "--key-list", str(shard_list),
        "--expected-count", str(len(selected)),
        "--expected-points", "50000",
        "--patch-size", str(args.patch_size),
        "--seed-k", str(args.seed_k),
        "--seed-k-alpha", str(args.seed_k_alpha),
        "--num-modules", "4",
        "--frame-knn", "32",
        "--noise-decay", "4",
        "--seed", str(args.seed),
        "--merge-strategy", args.merge_strategy,
        "--softmax-temperature", str(args.softmax_temperature),
        "--device", args.device,
    ]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
