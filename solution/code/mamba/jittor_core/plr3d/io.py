"""Canonical Track2 I/O, manifests and Jittor checkpoint loading."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import jittor as jt
import numpy as np


EXPECTED_POINTS = 50000
EXPECTED_CHANNELS = 3


class PLRError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_keys(path: Path) -> List[str]:
    keys = [
        line.strip().replace("\\", "/").strip("/")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not keys or len(keys) != len(set(keys)):
        raise PLRError(f"key list must be non-empty and unique: {path}")
    for key in keys:
        parts = key.split("/")
        if len(parts) != 3 or parts[0] != "shapenet":
            raise PLRError(f"non-canonical key: {key}")
    return keys


def canonical_key(path: Path, root: Path, filename: str) -> str:
    relative = path.relative_to(root)
    parts = relative.parts
    try:
        index = parts.index("shapenet")
    except ValueError as exc:
        raise PLRError(f"path has no shapenet component: {path}") from exc
    key_parts = parts[index : index + 3]
    if len(key_parts) != 3 or parts[-1] != filename or index + 4 != len(parts):
        raise PLRError(f"non-canonical path: {relative.as_posix()}")
    return "/".join(key_parts)


def discover_clouds(
    root: Path,
    filename: str,
    key_list: Optional[Path] = None,
) -> List[Tuple[str, Path]]:
    if key_list is not None:
        records = []
        for key in read_keys(key_list):
            path = root.joinpath(*key.split("/"), filename)
            if not path.is_file():
                raise PLRError(f"missing {filename} for {key}: {path}")
            records.append((key, path))
        return records
    records = [
        (canonical_key(path, root, filename), path)
        for path in sorted(root.rglob(filename))
    ]
    if not records:
        raise PLRError(f"no {filename} files found below {root}")
    keys = [key for key, _ in records]
    if len(keys) != len(set(keys)):
        raise PLRError(f"duplicate canonical keys below {root}")
    return records


def load_cloud(
    path: Path,
    expected_points: Optional[int] = EXPECTED_POINTS,
) -> np.ndarray:
    if not path.is_file():
        raise PLRError(f"missing cloud: {path}")
    array = np.load(path, allow_pickle=False)
    if array.dtype != np.float32 or array.ndim != 2 or array.shape[1] != 3:
        raise PLRError(f"invalid cloud {path}: dtype={array.dtype}, shape={array.shape}")
    if expected_points is not None and array.shape[0] != expected_points:
        raise PLRError(
            f"unexpected point count for {path}: {array.shape[0]} != {expected_points}"
        )
    if not np.isfinite(array).all():
        raise PLRError(f"non-finite cloud: {path}")
    return np.ascontiguousarray(array)


def save_cloud_new(path: Path, array: np.ndarray) -> None:
    value = np.ascontiguousarray(array, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 3 or not np.isfinite(value).all():
        raise PLRError(f"refusing invalid output array for {path}")
    if path.exists():
        raise PLRError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise PLRError(f"stale temporary output: {temporary}")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(temporary, path)


def write_json_new(path: Path, payload: object) -> None:
    if path.exists():
        raise PLRError(f"refusing to overwrite JSON: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def normalize_unit_sphere(points: jt.Var) -> Tuple[jt.Var, jt.Var, jt.Var]:
    point_max = points.max(dim=0, keepdims=True)
    point_min = points.min(dim=0, keepdims=True)
    center = (point_max + point_min) / 2.0
    centered = points - center
    scale = jt.sqrt((centered ** 2).sum(dim=-1)).max().reshape(1, 1)
    if not bool(jt.isfinite(scale).all().item()) or float(scale.item()) <= 0.0:
        raise PLRError("point cloud has invalid normalization scale")
    return centered / scale, center, scale


def checkpoint_state(payload: object) -> Tuple[Dict[str, object], Dict[str, object]]:
    if not isinstance(payload, dict):
        raise PLRError("checkpoint must be a dictionary")
    if "state_dict" in payload:
        state = payload["state_dict"]
        metadata = payload.get("metadata", {})
    else:
        state = payload
        metadata = {}
    if not isinstance(state, dict):
        raise PLRError("checkpoint state_dict must be a dictionary")
    state = {
        str(key): value
        for key, value in state.items()
        if not str(key).endswith("num_batches_tracked")
    }
    return state, metadata if isinstance(metadata, dict) else {}


def load_model_checkpoint(model, path: Path) -> Dict[str, object]:
    if not path.is_file():
        raise PLRError(f"missing checkpoint: {path}")
    payload = jt.load(str(path))
    state, metadata = checkpoint_state(payload)
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    shape_mismatch = []
    for key in sorted(set(expected) & set(state)):
        provided_shape = tuple(getattr(state[key], "shape", np.asarray(state[key]).shape))
        if tuple(expected[key].shape) != provided_shape:
            shape_mismatch.append((key, tuple(expected[key].shape), provided_shape))
    if missing or unexpected or shape_mismatch:
        raise PLRError(
            "checkpoint/model mismatch: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}, "
            f"shape_mismatch={shape_mismatch[:3]}"
        )
    model.load_parameters(state)
    return {
        "checkpoint_sha256": sha256_file(path),
        "state_key_count": len(state),
        "metadata": metadata,
    }


def save_jittor_checkpoint_new(
    model,
    path: Path,
    metadata: Dict[str, object],
) -> None:
    if path.exists():
        raise PLRError(f"refusing to overwrite checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    jt.save({"state_dict": model.state_dict(), "metadata": metadata}, str(temporary))
    os.replace(temporary, path)
