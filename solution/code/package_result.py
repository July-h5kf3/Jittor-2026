#!/usr/bin/env python3
"""Package a 200-item fused test tree as the official B-board result.zip."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-root", type=Path, required=True)
    parser.add_argument("--zip-path", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=200)
    args = parser.parse_args()

    expected = [
        "shapenet/00000000/btest_%06d/denoised.npy" % index
        for index in range(1, args.expected_count + 1)
    ]
    args.zip_path.parent.mkdir(parents=True, exist_ok=True)
    if args.zip_path.exists():
        raise RuntimeError("refusing to overwrite %s" % args.zip_path)
    with zipfile.ZipFile(args.zip_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in expected:
            source = args.pred_root.joinpath(*Path(name).parts)
            if not source.is_file():
                raise FileNotFoundError(source)
            value = np.load(source, allow_pickle=False)
            if value.shape != (50000, 3) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError("invalid prediction: %s" % source)
            archive.write(source, arcname=name)
    with zipfile.ZipFile(args.zip_path) as archive:
        names = archive.namelist()
        if names != expected:
            raise ValueError("ZIP member list mismatch")
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC check failed")
        for name in expected:
            payload = archive.read(name)
            source = args.pred_root.joinpath(*Path(name).parts)
            if payload != source.read_bytes():
                raise ValueError("ZIP bytes mismatch: %s" % name)
            value = np.load(io.BytesIO(payload), allow_pickle=False)
            if value.shape != (50000, 3) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError("invalid ZIP payload: %s" % name)
    audit = {
        "status": "passed",
        "zip_sha256": sha256_file(args.zip_path),
        "member_count": len(expected),
        "only_expected_members": True,
        "ordered_expected_members": True,
        "root": "shapenet/",
        "first_member": expected[0],
        "last_member": expected[-1],
        "shape": [50000, 3],
        "dtype": "float32",
        "finite": True,
        "crc_test": True,
        "zip_size_bytes": args.zip_path.stat().st_size,
    }
    args.audit_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
