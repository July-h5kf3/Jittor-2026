#!/usr/bin/env python3
"""B-board 76.999 recipe: median(48k, 60k, ipfn1024) - 0.01*(vm_fixed - noisy)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_keys(path: Path, expected: int) -> list[str]:
    keys = [
        line.strip().replace("\\", "/").strip("/")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(keys) != expected or len(set(keys)) != expected:
        raise ValueError("key count %d != %d" % (len(keys), expected))
    return keys


def find_npy(root: Path, key: str, name: str) -> Path:
    options = [
        root.joinpath(*key.split("/"), name),
        root.joinpath("shapenet", *key.split("/"), name),
    ]
    if key.startswith("shapenet/"):
        rest = key.split("/", 1)[1]
        options.append(root.joinpath(*rest.split("/"), name))
        options.append(root.joinpath("shapenet", *rest.split("/"), name))
    for path in options:
        if path.is_file():
            return path
    raise FileNotFoundError("%s / %s / %s" % (root, key, name))


def load_cloud(path: Path) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    if value.dtype != np.float32:
        value = np.asarray(value, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 3 or not np.isfinite(value).all():
        raise ValueError("invalid cloud: %s" % path)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noisy-root", type=Path, required=True)
    parser.add_argument("--key-list", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=200)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mamba-48k", type=Path, required=True)
    parser.add_argument("--mamba-60k", type=Path, required=True)
    parser.add_argument("--ipfn-1024", type=Path, required=True)
    parser.add_argument("--vm-fixed", type=Path, required=True)
    args = parser.parse_args()

    keys = load_keys(args.key_list, args.expected_count)
    args.output_root.mkdir(parents=True, exist_ok=True)
    for key in keys:
        noisy = load_cloud(find_npy(args.noisy_root, key, "noisy.npy"))
        a = load_cloud(find_npy(args.mamba_48k, key, "denoised.npy"))
        b = load_cloud(find_npy(args.mamba_60k, key, "denoised.npy"))
        c = load_cloud(find_npy(args.ipfn_1024, key, "denoised.npy"))
        vf = load_cloud(find_npy(args.vm_fixed, key, "denoised.npy"))
        for name, arr in (("48k", a), ("60k", b), ("i1", c), ("vf", vf)):
            if arr.shape != noisy.shape:
                raise ValueError("shape mismatch %s %s" % (name, key))
        med = np.median(np.stack([a, b, c], axis=0), axis=0).astype(np.float64)
        fused = (med - 0.01 * (vf.astype(np.float64) - noisy.astype(np.float64))).astype(
            np.float32
        )
        dest_key = key if key.startswith("shapenet/") else "shapenet/" + key
        out = args.output_root.joinpath(*dest_key.split("/"), "denoised.npy")
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, fused)
    manifest = {
        "schema": "fusion-global-m3i1-vf-n01-v1",
        "formula": "median(48k,60k,ipfn1024)-0.01*(vm_fixed-noisy)",
        "count": len(keys),
        "val144_total": 76.99905163980054,
        "submitted_result_zip": "result_fusion_global_m3i1_vf_n01_76p999.zip",
        "submitted_result_zip_sha256": "c8a6051a8274b9d52726ab57b5e34cd454dd36564036de4c6eb2d70004434da8",
    }
    (args.output_root / "fusion_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
