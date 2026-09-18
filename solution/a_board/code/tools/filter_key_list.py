#!/usr/bin/env python3
"""Filter a canonical key list by ShapeNet synset without reading point arrays."""

import argparse
from pathlib import Path


class FilterError(RuntimeError):
    pass


def read_keys(path: Path) -> list[str]:
    keys = [
        line.strip().replace("\\", "/").strip("/")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not keys or len(keys) != len(set(keys)):
        raise FilterError(f"key list is empty or duplicated: {path}")
    if any(len(key.split("/")) != 3 or not key.startswith("shapenet/") for key in keys):
        raise FilterError(f"invalid key list: {path}")
    return keys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--category", action="append", required=True)
    parser.add_argument("--expected-count", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FilterError(f"output exists: {args.output}")
    categories = set(args.category)
    keys = [key for key in read_keys(args.input) if key.split("/")[1] in categories]
    if not keys or len(keys) != len(set(keys)):
        raise FilterError("filtered key list is empty or duplicated")
    if args.expected_count is not None and len(keys) != args.expected_count:
        raise FilterError(f"expected {args.expected_count} keys, found {len(keys)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(keys) + "\n", encoding="utf-8")
    print(f"wrote {len(keys)} keys to {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FilterError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
