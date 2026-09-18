#!/usr/bin/env python3
"""Create an audited Track2 prediction ZIP from a canonical output tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

from plr3d.io import PLRError, load_cloud, read_keys, sha256_file, write_json_new


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--key-list", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=200)
    parser.add_argument("--expected-points", type=int, default=50000)
    args = parser.parse_args()

    if args.zip.exists() or args.audit.exists():
        raise PLRError("ZIP or audit target already exists")
    keys = read_keys(args.key_list)
    if len(keys) != args.expected_count:
        raise PLRError(f"expected {args.expected_count} keys, found {len(keys)}")
    entries = []
    args.zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for key in sorted(keys):
            path = args.root.joinpath(*key.split("/"), "denoised.npy")
            value = load_cloud(path, expected_points=args.expected_points)
            name = key + "/denoised.npy"
            archive.write(path, name)
            entries.append(
                {
                    "key": key,
                    "archive_name": name,
                    "file_sha256": sha256_file(path),
                    "array_sha256": hashlib.sha256(value.tobytes()).hexdigest(),
                }
            )
    with zipfile.ZipFile(args.zip, "r") as archive:
        names = archive.namelist()
        expected_names = [key + "/denoised.npy" for key in sorted(keys)]
        if names != expected_names or len(names) != len(set(names)):
            raise PLRError("archive members are not canonical, sorted and unique")
        for entry in entries:
            if hashlib.sha256(archive.read(entry["archive_name"])).hexdigest() != entry["file_sha256"]:
                raise PLRError(f"ZIP member mismatch: {entry['archive_name']}")
    payload = {
        "count": len(entries),
        "dtype": "float32",
        "shape": [args.expected_points, 3],
        "zip": str(args.zip.resolve()),
        "zip_size": args.zip.stat().st_size,
        "zip_sha256": sha256_file(args.zip),
        "entries": entries,
    }
    write_json_new(args.audit, payload)
    print(json.dumps({key: payload[key] for key in ("count", "zip_size", "zip_sha256")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PLRError as exc:
        print(f"ERROR: {exc}", flush=True)
        raise SystemExit(2)
