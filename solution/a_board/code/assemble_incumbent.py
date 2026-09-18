#!/usr/bin/env python3
"""Build the frozen MBI-009 standard-gated incumbent used by the final graph."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

from plr3d.io import PLRError, load_cloud, read_keys, save_cloud_new, sha256_file, write_json_new
from plr3d.routing import cloud_path, standard_route


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-list", type=Path, required=True)
    parser.add_argument("--noisy-root", type=Path, required=True)
    parser.add_argument("--rot-root", type=Path, required=True)
    parser.add_argument("--member-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=200)
    parser.add_argument("--expected-points", type=int, default=50000)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.output_root.exists() or args.manifest.exists():
        raise PLRError("incumbent output or manifest already exists")
    keys = read_keys(args.key_list)
    if len(keys) != args.expected_count:
        raise PLRError(f"expected {args.expected_count} keys, found {len(keys)}")
    temporary = args.output_root.with_name(args.output_root.name + ".tmp")
    entries = []
    selected_count = 0
    try:
        for key in keys:
            noisy_path = cloud_path(args.noisy_root, key, "noisy.npy")
            rot_path = cloud_path(args.rot_root, key)
            member_path = cloud_path(args.member_root, key)
            noisy = load_cloud(noisy_path, expected_points=args.expected_points)
            rot = load_cloud(rot_path, expected_points=args.expected_points)
            member = load_cloud(member_path, expected_points=args.expected_points)
            output, ratio, selected = standard_route(noisy, rot, member, 4.0)
            selected_count += int(selected)
            target = cloud_path(temporary, key)
            save_cloud_new(target, output)
            entries.append(
                {
                    "key": key,
                    "category": key.split("/")[1],
                    "ratio_p95": ratio,
                    "selected": selected,
                    "noisy_sha256": sha256_file(noisy_path),
                    "rot_sha256": sha256_file(rot_path),
                    "member_sha256": sha256_file(member_path),
                    "output_sha256": sha256_file(target),
                }
            )
        os.replace(temporary, args.output_root)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    payload = {
        "experiment": "NKAI-MBI009-equivalent-final-incumbent",
        "framework": "Jittor",
        "count": len(entries),
        "selected_count": selected_count,
        "category_counts": dict(Counter(item["category"] for item in entries)),
        "formula": "q95(||member-noisy||/max(||ROT-noisy||,tiny))<=4 ? float32(.25*ROT+.75*member) : ROT",
        "dead_branch_pruning": "Later Full/PLR routes overwrite historical MBI020-only branches; the retained MBI009 incumbent is identical on the three final inherited non-tail clouds.",
        "entries": entries,
    }
    write_json_new(args.manifest, payload)
    print(json.dumps({"count": len(entries), "selected_count": selected_count}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PLRError as exc:
        print(f"ERROR: {exc}", flush=True)
        raise SystemExit(2)
