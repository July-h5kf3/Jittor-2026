#!/usr/bin/env python3
"""Rebuild the frozen Full DCR002/TSD004/SSD010 noisy-only route."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from plr3d.io import PLRError, load_cloud, read_keys, save_cloud_new, sha256_file, write_json_new
from plr3d.routing import SOFA, TABLE, cloud_path, extrapolate, standard_route

DCD_DIRECT = frozenset({"02691156", "03046257", "03642806", "04330267", "04468005"})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-list", type=Path, required=True)
    parser.add_argument("--noisy-root", type=Path, required=True)
    parser.add_argument("--rot-root", type=Path, required=True)
    parser.add_argument("--incumbent-root", type=Path, required=True)
    parser.add_argument("--dcd-root", type=Path, required=True)
    parser.add_argument("--table-root", type=Path, required=True)
    parser.add_argument("--sofa-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=200)
    args = parser.parse_args()
    if args.output_root.exists() or args.manifest.exists():
        raise PLRError("Full output or manifest already exists")
    keys = read_keys(args.key_list)
    if len(keys) != args.expected_count:
        raise PLRError(f"expected {args.expected_count} keys, found {len(keys)}")

    temporary = args.output_root.with_name(args.output_root.name + ".tmp")
    entries = []
    try:
        for key in keys:
            category = key.split("/")[1]
            incumbent_path = cloud_path(args.incumbent_root, key)
            incumbent = load_cloud(incumbent_path)
            details = {}
            if category in DCD_DIRECT or category == TABLE or category == SOFA:
                noisy = load_cloud(cloud_path(args.noisy_root, key, "noisy.npy"))
                rot = load_cloud(cloud_path(args.rot_root, key))
                if category in DCD_DIRECT:
                    member_path = cloud_path(args.dcd_root, key)
                    output, ratio, selected = standard_route(
                        noisy, rot, load_cloud(member_path)
                    )
                    route = "dcd"
                    details = {"ratio": ratio, "selected": selected}
                elif category == TABLE:
                    member_path = cloud_path(args.table_root, key)
                    output, ratio, selected = standard_route(
                        noisy, rot, load_cloud(member_path)
                    )
                    route = "table"
                    details = {"ratio": ratio, "selected": selected}
                else:
                    dcd_path = cloud_path(args.dcd_root, key)
                    sofa_path = cloud_path(args.sofa_root, key)
                    source, dcd_ratio, dcd_selected = standard_route(
                        noisy, rot, load_cloud(dcd_path)
                    )
                    strong, sofa_ratio, sofa_selected = standard_route(
                        noisy, rot, load_cloud(sofa_path)
                    )
                    output = extrapolate(source, strong, 1.25)
                    route = "sofa"
                    details = {
                        "dcd_ratio": dcd_ratio,
                        "dcd_selected": dcd_selected,
                        "sofa_ratio": sofa_ratio,
                        "sofa_selected": sofa_selected,
                    }
                destination = cloud_path(temporary, key)
                save_cloud_new(destination, output)
            else:
                route = "inherit"
                destination = cloud_path(temporary, key)
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(incumbent_path, destination)
                except OSError:
                    shutil.copy2(incumbent_path, destination)
            entries.append(
                {
                    "key": key,
                    "category": category,
                    "route": route,
                    "incumbent_sha256": sha256_file(incumbent_path),
                    "output_sha256": sha256_file(destination),
                    "details": details,
                }
            )
        os.replace(temporary, args.output_root)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    payload = {
        "experiment": "Full-DCR002-TSD004-SSD010-Jittor-rebuild",
        "framework": "Jittor inference + Numpy noisy-only route",
        "pytorch_runtime": False,
        "count": len(entries),
        "route_counts": dict(Counter(entry["route"] for entry in entries)),
        "dcd_direct_categories": sorted(DCD_DIRECT),
        "formula": {
            "standard": "relative-p95 gate; .25*ROT+.75*member else ROT",
            "sofa": "DCD_standard+1.25*(SSD_standard-DCD_standard)",
            "inherit": "byte-exact incumbent",
        },
        "entries": entries,
    }
    write_json_new(args.manifest, payload)
    print(json.dumps({"count": len(entries), "route_counts": payload["route_counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PLRError as exc:
        print(f"ERROR: {exc}", flush=True)
        raise SystemExit(2)
