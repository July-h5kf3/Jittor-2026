#!/usr/bin/env python
"""Pre-warm the parsed-mesh cache used by ObjLazyAsset.

Parsing ShapeNet .obj text on every epoch dominates dataloader time
(~55 ms/item). ObjLazyAsset.load() transparently caches the parsed mesh to
`<obj>.cache.npz`; this script builds that cache up-front in parallel so the
first training epoch is already fast.

Usage:
    python scripts/precache_meshes.py \
        --datalist datalist/train.txt datalist/validate.txt \
        --root dataset_train \
        --data-name models/model_normalized.obj \
        --workers 48
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# make `src` importable when run from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.datapath import ObjLazyAsset


def _build_one(path: str) -> tuple[str, bool, str]:
    try:
        asset = ObjLazyAsset(path=path)
        cache = asset._cache_path()
        if os.path.exists(cache):
            return path, True, "cached"
        asset._load_mesh()  # parses + writes the cache as a side effect
        return path, True, "built"
    except Exception as e:  # noqa: BLE001 - report and keep going
        return path, False, repr(e)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datalist", nargs="+", required=True)
    parser.add_argument("--root", default="dataset_train")
    parser.add_argument("--data-name", default="models/model_normalized.obj")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    args = parser.parse_args()

    rels: list[str] = []
    seen: set[str] = set()
    for dl in args.datalist:
        with open(dl, "r", encoding="utf-8") as f:
            for line in f:
                rel = line.strip()
                if rel and rel not in seen:
                    seen.add(rel)
                    rels.append(rel)

    paths = [os.path.join(args.root, rel, args.data_name) for rel in rels]
    total = len(paths)
    print(f"pre-caching {total} meshes with {args.workers} workers ...", flush=True)

    t0 = time.perf_counter()
    built = cached = failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_build_one, p): p for p in paths}
        for i, fut in enumerate(as_completed(futures), 1):
            path, ok, msg = fut.result()
            if not ok:
                failed += 1
                print(f"  [FAIL] {path}: {msg}", flush=True)
            elif msg == "built":
                built += 1
            else:
                cached += 1
            if i % 500 == 0 or i == total:
                dt = time.perf_counter() - t0
                print(
                    f"  {i}/{total}  built={built} cached={cached} "
                    f"failed={failed}  ({dt:.1f}s)",
                    flush=True,
                )

    dt = time.perf_counter() - t0
    print(f"done: built={built} cached={cached} failed={failed} in {dt:.1f}s", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
