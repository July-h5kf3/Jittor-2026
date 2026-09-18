#!/usr/bin/env python3
"""Assemble the frozen PLR-003 noisy-only route from Jittor predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plr3d.io import PLRError, read_keys
from plr3d.routing import assemble_plr003


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-list", type=Path, required=True)
    parser.add_argument("--noisy-root", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--rot-root", type=Path, required=True)
    parser.add_argument("--full-dcd-root", type=Path, required=True)
    parser.add_argument("--airplane-root", type=Path, required=True)
    parser.add_argument("--table-root", type=Path, required=True)
    parser.add_argument("--sofa-root", type=Path, required=True)
    parser.add_argument("--tail-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=200)
    parser.add_argument("--expected-points", type=int, default=50000)
    args = parser.parse_args()

    keys = read_keys(args.key_list)
    if len(keys) != args.expected_count:
        raise PLRError(f"expected {args.expected_count} keys, found {len(keys)}")
    payload = assemble_plr003(
        keys=keys,
        noisy_root=args.noisy_root,
        full_root=args.full_root,
        rot_root=args.rot_root,
        full_dcd_root=args.full_dcd_root,
        airplane_root=args.airplane_root,
        table_root=args.table_root,
        sofa_root=args.sofa_root,
        tail_root=args.tail_root,
        output_root=args.output_root,
        manifest_path=args.manifest,
        expected_points=args.expected_points,
    )
    print(
        json.dumps(
            {
                "count": payload["count"],
                "route_counts": payload["route_counts"],
                "changed_vs_full": payload["changed_vs_full"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PLRError as exc:
        print(f"ERROR: {exc}", flush=True)
        raise SystemExit(2)
