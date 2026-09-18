#!/usr/bin/env python3
"""Merge disjoint canonical inference shards without copying predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", action="append", type=Path, required=True)
    parser.add_argument("--expected-key-list", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expected = [
        line.strip().replace("\\", "/").strip("/")
        for line in args.expected_key_list.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("expected key list must be non-empty and unique")
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    temporary = args.output_root.with_name(args.output_root.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.mkdir(parents=True)
    entries = {}
    shard_manifests = []
    try:
        for shard_raw in args.shard:
            shard = shard_raw.resolve(strict=True)
            manifest_path = shard / "inference_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            shard_manifests.append({
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
                "count": int(manifest["count"]),
            })
            for entry in manifest["entries"]:
                key = entry["key"]
                if key in entries:
                    raise ValueError("duplicate key across inference shards: %s" % key)
                source = shard.joinpath(*key.split("/"), "denoised.npy")
                if not source.is_file() or sha256_file(source) != entry["output_file_sha256"]:
                    raise ValueError("inference shard prediction SHA mismatch: %s" % key)
                target = temporary.joinpath(*key.split("/"), "denoised.npy")
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source, target)
                except OSError:
                    shutil.copy2(source, target)
                entries[key] = entry
        if set(entries) != set(expected):
            raise ValueError(
                "inference shard coverage mismatch: missing=%s unexpected=%s" %
                (sorted(set(expected) - set(entries))[:5], sorted(set(entries) - set(expected))[:5])
            )
        audit = {
            "schema": "3dmambaipf-b-inference-shard-merge-v1",
            "count": len(expected),
            "expected_key_list": str(args.expected_key_list.resolve()),
            "expected_key_list_sha256": sha256_file(args.expected_key_list),
            "shards": shard_manifests,
            "entries": [entries[key] for key in expected],
        }
        with (temporary / "merge_manifest.json").open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            handle.write(
                json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
            )
        os.replace(temporary, args.output_root)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    print(json.dumps({"count": len(expected), "output_root": str(args.output_root.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
