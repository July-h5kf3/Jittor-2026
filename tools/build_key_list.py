#!/usr/bin/env python3
"""Build a sorted shapenet/<synset>/<model> key list from an external dataset."""

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--filename", choices=("noisy.npy", "clean.npy"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    files = sorted(args.root.glob(f"shapenet/*/*/{args.filename}"))
    keys = [path.parent.relative_to(args.root).as_posix() for path in files]
    if not keys or len(keys) != len(set(keys)):
        raise RuntimeError("dataset discovery produced an empty or duplicate key list")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(keys) + "\n", encoding="utf-8")
    print(f"wrote {len(keys)} keys to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
