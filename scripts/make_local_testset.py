#!/usr/bin/env python
"""Build a LOCAL validation test set with ground truth, mirroring the hidden
competition test format, so we can score denoisers offline with evaluate.py.

For each model listed in a datalist (default: datalist/validate.txt -- held-out
meshes), it:
  1. loads the mesh, samples N clean surface points,
  2. normalizes to the unit sphere (same convention as the real test data),
  3. adds Gaussian noise of std `noise_std` (default 0.015, which reproduces the
     ~1.04 radius observed in the real test clouds),
  4. writes clean.npy (GT) and noisy.npy under <out>/<rel>/.

Then run inference (predict) pointed at the noisy clouds, and score with:
  python evaluate.py --pred_dir <results> --gt_dir <out> --noisy_dir <out> \
      --mesh_dir ./dataset_train --workers 16

Usage:
  python scripts/make_local_testset.py --datalist datalist/validate.txt \
      --root dataset_train --out localtest --num_points 50000 \
      --noise_std 0.015 --limit 40 --seed 2026
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.datapath import ObjLazyAsset
from src.data.utils import sample_vertex_groups


def unit_sphere_transform(pc: np.ndarray):
    """Return (center, scale) mapping pc to the unit sphere."""
    center = (pc.max(0) + pc.min(0)) / 2.0
    scale = np.sqrt(((pc - center) ** 2).sum(1).max()) + 1e-12
    return center, scale


def write_obj(path: str, verts: np.ndarray, faces: np.ndarray):
    """Write a minimal OBJ (v + f, 1-indexed) so evaluate.py/pcu can load it."""
    with open(path, "w") as fh:
        lines = [f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in verts]
        lines += [f"f {a+1} {b+1} {c+1}\n" for a, b, c in faces]
        fh.writelines(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datalist", default="datalist/validate.txt")
    ap.add_argument("--root", default="dataset_train")
    ap.add_argument("--data-name", default="models/model_normalized.obj")
    ap.add_argument("--out", default="localtest")
    ap.add_argument("--num_points", type=int, default=50000)
    ap.add_argument("--num_vertex_samples", type=int, default=1024)
    ap.add_argument("--noise_std", type=float, default=0.015)
    ap.add_argument("--noise", choices=["gaussian", "laplace"], default="gaussian")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    rels = [l.strip() for l in open(args.datalist) if l.strip()]
    if args.limit:
        rels = rels[: args.limit]

    ok = fail = 0
    for i, rel in enumerate(rels):
        obj = os.path.join(args.root, rel, args.data_name)
        try:
            v, f = ObjLazyAsset(path=obj)._load_mesh()
            v = np.asarray(v, dtype=np.float64)
            f = np.asarray(f)
            clean_raw, _, _, _ = sample_vertex_groups(
                vertices=v, faces=f,
                num_samples=args.num_points, num_vertex_samples=args.num_vertex_samples,
            )
            clean_raw = np.asarray(clean_raw, dtype=np.float64)
            # Normalize clean to the unit sphere (so the denoiser sees the same
            # scale as the real test). Apply the SAME transform to the mesh so
            # P2S stays in a consistent frame.
            center, scale = unit_sphere_transform(clean_raw)
            clean = (clean_raw - center) / scale
            v_n = (v - center) / scale
            if args.noise == "gaussian":
                noise = rng.normal(0.0, args.noise_std, size=clean.shape)
            else:
                noise = rng.laplace(0.0, args.noise_std, size=clean.shape)
            noisy = clean + noise
            d = os.path.join(args.out, rel)
            os.makedirs(os.path.join(d, "models"), exist_ok=True)
            np.save(os.path.join(d, "clean.npy"), clean.astype(np.float32))
            np.save(os.path.join(d, "noisy.npy"), noisy.astype(np.float32))
            write_obj(os.path.join(d, "models", "model_normalized.obj"), v_n, f)
            ok += 1
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"  [FAIL] {rel}: {e!r}", flush=True)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(rels)} ok={ok} fail={fail}", flush=True)
    print(f"done: ok={ok} fail={fail} -> {args.out}  (N={args.num_points}, "
          f"{args.noise} std={args.noise_std})", flush=True)
    # also emit a datalist of the generated samples for the predict step
    with open(os.path.join(args.out, "local.txt"), "w") as fh:
        for rel in rels:
            if os.path.exists(os.path.join(args.out, rel, "noisy.npy")):
                fh.write(rel + "\n")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
