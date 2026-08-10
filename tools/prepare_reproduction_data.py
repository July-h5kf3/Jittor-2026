#!/usr/bin/env python3
"""Build deterministic clean-surface datasets used by the frozen training chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np


OFFICIAL_FREQUENCY_3000 = {
    "04379243": 1401,
    "02691156": 533,
    "04256520": 457,
    "04401088": 198,
    "03046257": 137,
    "02871439": 76,
    "03642806": 61,
    "04468005": 61,
    "02876657": 46,
    "04330267": 30,
}
TAIL_1885 = {
    "02871439": 220,
    "02876657": 250,
    "03046257": 350,
    "03642806": 190,
    "04330267": 55,
    "04401088": 650,
    "04468005": 170,
}
PRESET_COUNTS: Dict[str, Optional[Dict[str, int]]] = {
    "base": None,
    "mbi009_650": {},  # filled as 50 per discovered category
    "dcd001_3000": OFFICIAL_FREQUENCY_3000,
    "table_1200": {"04379243": 1200},
    "airplane_1000": {"02691156": 1000},
    "sofa_1000": {"04256520": 1000},
    "tail_1885": TAIL_1885,
}
PRESET_SEEDS = {
    "base": (20202020, 20202021),
    "mbi009_650": (20260812, 20260813),
    "dcd001_3000": (20261301, 20261302),
    "table_1200": (20260901, 20260902),
    "airplane_1000": (20260914, 20260915),
    "sofa_1000": (20260924, 20260925),
    "tail_1885": (20262721, 20262722),
}
PRESET_SURFACE_NAMESPACE = {
    "base": "nkai-base-surface",
    "mbi009_650": "mbi009-surface",
    "dcd001_3000": "dcd001-surface",
    "table_1200": "mbi011-category-surface",
    "airplane_1000": "mbi012-train-surface",
    "sofa_1000": "mbi012-train-surface",
    "tail_1885": "nkai-tail-seed-map-required",
}


class PrepareError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_digest(namespace: str, seed: int, key: str) -> bytes:
    return hashlib.sha256(f"{namespace}:{seed}:{key}".encode("utf-8")).digest()


def stable_seed(namespace: str, seed: int, key: str) -> int:
    return int.from_bytes(stable_digest(namespace, seed, key)[:4], "little")


def read_keys(path: Path) -> list[str]:
    keys = [line.strip().replace("\\", "/").strip("/") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not keys or len(keys) != len(set(keys)):
        raise PrepareError(f"key list must be non-empty and unique: {path}")
    for key in keys:
        parts = key.split("/")
        if len(parts) != 3 or parts[0] != "shapenet":
            raise PrepareError(f"invalid key: {key}")
    return keys


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    cache = Path(str(path) + ".cache.npz")
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as payload:
            vertices = np.asarray(payload["vertices"], dtype=np.float64)
            faces = np.asarray(payload["faces"], dtype=np.int64)
    else:
        vertex_rows = []
        face_rows = []
        for raw in path.read_text(encoding="utf-8", errors="strict").splitlines():
            line = raw.strip()
            if line.startswith("v "):
                values = line.split()[1:4]
                if len(values) != 3:
                    raise PrepareError(f"invalid OBJ vertex: {path}")
                vertex_rows.append([float(value) for value in values])
            elif line.startswith("f "):
                indices = []
                for token in line.split()[1:]:
                    raw_index = int(token.split("/", 1)[0])
                    index = raw_index - 1 if raw_index > 0 else len(vertex_rows) + raw_index
                    indices.append(index)
                if len(indices) < 3:
                    raise PrepareError(f"invalid OBJ face: {path}")
                for offset in range(1, len(indices) - 1):
                    face_rows.append([indices[0], indices[offset], indices[offset + 1]])
        vertices = np.asarray(vertex_rows, dtype=np.float64)
        faces = np.asarray(face_rows, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise PrepareError(f"invalid mesh vertices: {path}")
    if faces.ndim != 2 or faces.shape[1] != 3 or not faces.size:
        raise PrepareError(f"invalid mesh faces: {path}")
    if faces.min() < 0 or faces.max() >= vertices.shape[0]:
        raise PrepareError(f"OBJ face index out of range: {path}")
    return vertices, faces


def sample_surface(vertices: np.ndarray, faces: np.ndarray, count: int, rng: np.random.RandomState) -> np.ndarray:
    edge0 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    edge1 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    cumulative = np.cumsum(np.linalg.norm(np.cross(edge0, edge1), axis=1), dtype=np.float64)
    if not np.isfinite(cumulative[-1]) or cumulative[-1] <= 0.0:
        raise PrepareError("mesh has no finite positive-area face")
    face_index = np.searchsorted(cumulative, rng.random_sample(count) * cumulative[-1])
    origins = vertices[faces[:, 0]][face_index]
    vectors = (vertices[faces[:, 1:]] - vertices[faces[:, 0]][:, None, :])[face_index]
    lengths = rng.random_sample((count, 2, 1))
    reflected = lengths.sum(axis=1).reshape(-1) > 1.0
    lengths[reflected] -= 1.0
    points = origins + (vectors * np.abs(lengths)).sum(axis=1)
    return np.ascontiguousarray(points.astype(np.float32))


def select_keys(preset: str, candidates: list[str], excluded: set[str]) -> list[str]:
    eligible = [key for key in candidates if key not in excluded]
    by_category: Dict[str, list[str]] = defaultdict(list)
    for key in eligible:
        by_category[key.split("/")[1]].append(key)
    selection_seed, _ = PRESET_SEEDS[preset]
    if preset == "base":
        return sorted(eligible)
    counts = PRESET_COUNTS[preset]
    if preset == "mbi009_650":
        if len(by_category) != 13:
            raise PrepareError(f"MBI009 expects 13 categories, found {len(by_category)}")
        counts = {category: 50 for category in by_category}
    assert counts is not None
    chosen = []
    for category in sorted(counts):
        required = counts[category]
        pool = by_category.get(category, [])
        if len(pool) < required:
            raise PrepareError(f"preset {preset}: category {category} has {len(pool)}, requires {required}")
        ordered = sorted(pool, key=lambda key: (stable_digest(f"{preset}-{category}", selection_seed, key), key))
        chosen.extend(ordered[:required])
    if len(chosen) != sum(counts.values()) or len(chosen) != len(set(chosen)):
        raise PrepareError(f"invalid selection for {preset}")
    return sorted(chosen)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=tuple(PRESET_COUNTS), required=True)
    parser.add_argument("--candidate-list", type=Path, required=True)
    parser.add_argument(
        "--selected-list",
        type=Path,
        help="frozen training-key list; bypasses selection while retaining full validation",
    )
    parser.add_argument(
        "--seed-map-json",
        type=Path,
        help="optional JSON object mapping every selected key to its historical surface seed",
    )
    parser.add_argument("--exclude-list", type=Path, action="append", default=[])
    parser.add_argument("--mesh-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-list", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--point-count", type=int, default=50000)
    parser.add_argument("--mesh-name", default="models/model_normalized.obj")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    targets = (args.output_root, args.output_list, args.manifest)
    if any(path.exists() for path in targets):
        raise PrepareError("one or more output targets already exist")
    if args.point_count <= 0 or not args.mesh_root.is_dir():
        raise PrepareError("invalid point count or mesh root")
    candidates = read_keys(args.candidate_list)
    excluded = set()
    exclusion_records = []
    for path in args.exclude_list:
        values = read_keys(path)
        excluded.update(values)
        exclusion_records.append({"path": str(path.resolve()), "count": len(values), "sha256": sha256_file(path)})
    if args.selected_list is None:
        selected = select_keys(args.preset, candidates, excluded)
    else:
        selected = sorted(read_keys(args.selected_list))
        eligible = set(candidates) - excluded
        unknown = sorted(set(selected) - eligible)
        if unknown:
            raise PrepareError(f"selected list contains {len(unknown)} ineligible keys")
        expected_counts = PRESET_COUNTS[args.preset]
        actual_counts: Dict[str, int] = defaultdict(int)
        for key in selected:
            actual_counts[key.split("/")[1]] += 1
        if args.preset == "base":
            if set(selected) != eligible:
                raise PrepareError("base selected list must equal the eligible candidate inventory")
        elif args.preset == "mbi009_650":
            if set(actual_counts.values()) != {50} or len(actual_counts) != 13:
                raise PrepareError("MBI009 selected list must contain 50 keys from each of 13 categories")
        elif dict(actual_counts) != expected_counts:
            raise PrepareError(
                f"selected-list category counts mismatch: {dict(actual_counts)} != {expected_counts}"
            )
    _, surface_seed = PRESET_SEEDS[args.preset]
    seed_map = None
    if args.seed_map_json is not None:
        seed_payload = json.loads(args.seed_map_json.read_text(encoding="utf-8"))
        if not isinstance(seed_payload, dict):
            raise PrepareError("seed map must be a JSON object")
        seed_map = {str(key): int(value) for key, value in seed_payload.items()}
        if set(seed_map) != set(selected):
            raise PrepareError("seed map keys must exactly match selected keys")
    elif args.preset == "tail_1885":
        raise PrepareError("tail_1885 requires --seed-map-json for historical category seeds")
    temporary = args.output_root.with_name(args.output_root.name + ".building")
    if temporary.exists():
        raise PrepareError(f"stale temporary output: {temporary}")
    entries = []
    try:
        for index, key in enumerate(selected, 1):
            mesh = args.mesh_root.joinpath(*key.split("/"), *args.mesh_name.split("/"))
            if not mesh.is_file():
                raise FileNotFoundError(mesh)
            seed = (
                seed_map[key]
                if seed_map is not None
                else stable_seed(PRESET_SURFACE_NAMESPACE[args.preset], surface_seed, key)
            )
            vertices, faces = load_mesh(mesh)
            clean = sample_surface(vertices, faces, args.point_count, np.random.RandomState(seed))
            if clean.shape != (args.point_count, 3) or not np.isfinite(clean).all():
                raise PrepareError(f"invalid sampled cloud: {key}")
            target = temporary.joinpath(*key.split("/"), "clean.npy")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                np.save(handle, clean, allow_pickle=False)
            entries.append({"key": key, "mesh_sha256": sha256_file(mesh), "clean_sha256": sha256_file(target), "surface_seed": seed})
            if index % 100 == 0 or index == len(selected):
                print(f"{args.preset}: sampled {index}/{len(selected)}", flush=True)
        os.replace(temporary, args.output_root)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    args.output_list.parent.mkdir(parents=True, exist_ok=True)
    args.output_list.write_text("\n".join(selected) + "\n", encoding="utf-8")
    payload = {
        "experiment": f"NKAI-{args.preset}-deterministic-surface-data",
        "preset": args.preset,
        "candidate_list": {"path": str(args.candidate_list.resolve()), "sha256": sha256_file(args.candidate_list), "count": len(candidates)},
        "exclude_lists": exclusion_records,
        "selected_list": (
            None
            if args.selected_list is None
            else {
                "path": str(args.selected_list.resolve()),
                "sha256": sha256_file(args.selected_list),
                "count": len(selected),
            }
        ),
        "seed_map": (
            None
            if args.seed_map_json is None
            else {
                "path": str(args.seed_map_json.resolve()),
                "sha256": sha256_file(args.seed_map_json),
                "count": len(seed_map or {}),
            }
        ),
        "excluded_count": len(excluded),
        "mesh_root": str(args.mesh_root.resolve()),
        "mesh_name": args.mesh_name,
        "point_count": args.point_count,
        "output_root": str(args.output_root.resolve()),
        "train": {"path": str(args.output_list.resolve()), "sha256": sha256_file(args.output_list), "count": len(selected)},
        "entries": entries,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"preset": args.preset, "count": len(selected), "manifest": str(args.manifest)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PrepareError as exc:
        print(f"ERROR: {exc}", flush=True)
        raise SystemExit(2)
