"""Frozen noisy-only PLR-001 -> PLR-002 -> PLR-003 assembly."""

from __future__ import annotations

import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .io import PLRError, load_cloud, save_cloud_new, sha256_file, write_json_new


AIRPLANE = "02691156"
SOFA = "04256520"
TABLE = "04379243"
TAIL_CATEGORIES = frozenset(
    {
        "02871439",
        "02876657",
        "03046257",
        "03642806",
        "04330267",
        "04401088",
        "04468005",
    }
)


def cloud_path(root: Path, key: str, filename: str = "denoised.npy") -> Path:
    return root.joinpath(*key.split("/"), filename)


def standard_route(
    noisy: np.ndarray,
    rot: np.ndarray,
    member: np.ndarray,
    threshold: float = 4.0,
) -> Tuple[np.ndarray, float, bool]:
    noisy64 = noisy.astype(np.float64)
    rot64 = rot.astype(np.float64)
    member64 = member.astype(np.float64)
    numerator = np.linalg.norm(member64 - noisy64, axis=1)
    denominator = np.maximum(
        np.linalg.norm(rot64 - noisy64, axis=1),
        np.finfo(np.float64).tiny,
    )
    ratio = float(np.quantile(numerator / denominator, 0.95))
    selected = ratio <= threshold
    if selected:
        output = (0.25 * rot64 + 0.75 * member64).astype(np.float32)
    else:
        output = rot.copy()
    return np.ascontiguousarray(output), ratio, selected


def extrapolate(source: np.ndarray, strong: np.ndarray, alpha: float) -> np.ndarray:
    output = source.astype(np.float64) + alpha * (
        strong.astype(np.float64) - source.astype(np.float64)
    )
    return np.ascontiguousarray(output.astype(np.float32))


def _load(
    root: Path,
    key: str,
    filename: str = "denoised.npy",
    expected_points: int = 50000,
):
    path = cloud_path(root, key, filename)
    return load_cloud(path, expected_points=expected_points), path


def assemble_plr003(
    keys: List[str],
    noisy_root: Path,
    full_root: Path,
    rot_root: Path,
    full_dcd_root: Path,
    airplane_root: Path,
    table_root: Path,
    sofa_root: Path,
    tail_root: Path,
    output_root: Path,
    manifest_path: Path,
    expected_points: int = 50000,
    threshold: float = 4.0,
    sofa_alpha: float = 1.25,
    table_alpha: float = 1.25,
) -> Dict[str, object]:
    if threshold != 4.0 or sofa_alpha != 1.25 or table_alpha != 1.25:
        raise PLRError("PLR route constants are frozen at 4.0/1.25/1.25")
    roots = {
        "noisy": noisy_root,
        "full": full_root,
        "rot": rot_root,
        "full_dcd": full_dcd_root,
        "airplane": airplane_root,
        "table": table_root,
        "sofa": sofa_root,
        "tail": tail_root,
    }
    for label, root in roots.items():
        if not root.is_dir():
            raise PLRError(f"missing {label} root: {root}")
    if output_root.exists() or manifest_path.exists():
        raise PLRError("PLR output root or manifest already exists")

    temporary = output_root.with_name(output_root.name + ".tmp")
    if temporary.exists():
        raise PLRError(f"stale temporary output: {temporary}")
    entries = []
    try:
        for key in keys:
            category = key.split("/")[1]
            noisy, noisy_path = _load(
                noisy_root, key, "noisy.npy", expected_points
            )
            full, full_path = _load(full_root, key, expected_points=expected_points)
            rot, rot_path = _load(rot_root, key, expected_points=expected_points)
            if noisy.shape[0] != expected_points:
                raise PLRError(f"unexpected point count for {key}: {noisy.shape[0]}")
            details = {}

            if category == AIRPLANE:
                member, member_path = _load(
                    airplane_root, key, expected_points=expected_points
                )
                plr001, ratio, selected = standard_route(noisy, rot, member, threshold)
                details["plr001"] = {
                    "route": "airplane_standard",
                    "ratio": ratio,
                    "selected": selected,
                    "member_sha256": sha256_file(member_path),
                }
            elif category == TABLE:
                member, member_path = _load(
                    table_root, key, expected_points=expected_points
                )
                plr001, ratio, selected = standard_route(noisy, rot, member, threshold)
                details["plr001"] = {
                    "route": "table_standard",
                    "ratio": ratio,
                    "selected": selected,
                    "member_sha256": sha256_file(member_path),
                }
            elif category == SOFA:
                dcd_member, dcd_path = _load(
                    full_dcd_root, key, expected_points=expected_points
                )
                sofa_member, sofa_path = _load(
                    sofa_root, key, expected_points=expected_points
                )
                source, dcd_ratio, dcd_selected = standard_route(
                    noisy, rot, dcd_member, threshold
                )
                strong, sofa_ratio, sofa_selected = standard_route(
                    noisy, rot, sofa_member, threshold
                )
                plr001 = extrapolate(source, strong, sofa_alpha)
                details["plr001"] = {
                    "route": "sofa_extrapolate_1p25",
                    "dcd_ratio": dcd_ratio,
                    "dcd_selected": dcd_selected,
                    "sofa_ratio": sofa_ratio,
                    "sofa_selected": sofa_selected,
                    "dcd_member_sha256": sha256_file(dcd_path),
                    "member_sha256": sha256_file(sofa_path),
                }
            else:
                plr001 = full
                details["plr001"] = {"route": "full_fallback"}

            if category == TABLE:
                plr002 = extrapolate(full, plr001, table_alpha)
                details["plr002"] = {"route": "table_extrapolate_1p25"}
            else:
                plr002 = plr001
                details["plr002"] = {"route": "plr001_fallback"}

            if category in TAIL_CATEGORIES:
                tail_member, tail_path = _load(
                    tail_root, key, expected_points=expected_points
                )
                output, tail_ratio, tail_selected = standard_route(
                    noisy, rot, tail_member, threshold
                )
                final_route = "plr003_tail_standard"
                details["plr003"] = {
                    "route": final_route,
                    "ratio": tail_ratio,
                    "selected": tail_selected,
                    "member_sha256": sha256_file(tail_path),
                }
            else:
                output = plr002
                final_route = "plr002_fallback"
                details["plr003"] = {"route": final_route, "selected": False}

            if (
                output.dtype != np.float32
                or output.shape != noisy.shape
                or not np.isfinite(output).all()
            ):
                raise PLRError(f"invalid assembled output for {key}")
            destination = cloud_path(temporary, key)
            save_cloud_new(destination, output)
            entries.append(
                {
                    "key": key,
                    "category": category,
                    "route": final_route,
                    "noisy_sha256": sha256_file(noisy_path),
                    "full_sha256": sha256_file(full_path),
                    "rot_sha256": sha256_file(rot_path),
                    "output_sha256": sha256_file(destination),
                    "byte_equal_full": destination.read_bytes() == full_path.read_bytes(),
                    "details": details,
                }
            )
        os.replace(temporary, output_root)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    route_counts = dict(Counter(entry["route"] for entry in entries))
    payload = {
        "experiment": "PLR-003-Jittor-native-noisy-only-assembly",
        "count": len(entries),
        "clean_read": False,
        "mesh_read": False,
        "official_metrics_read": False,
        "tail_categories": sorted(TAIL_CATEGORIES),
        "formula": {
            "standard": "q95(||member-noisy||/max(||ROT-noisy||,tiny))<=4 ? float32(.25*ROT+.75*member) : ROT",
            "plr001_sofa": "DCD_standard + 1.25*(sofa_standard-DCD_standard)",
            "plr002_table": "Full + 1.25*(PLR001-Full)",
            "plr003_tail": "standard(noisy,ROT,tail); PLR002 fallback otherwise",
        },
        "route_counts": route_counts,
        "changed_vs_full": sum(not entry["byte_equal_full"] for entry in entries),
        "output_root": str(output_root.resolve()),
        "entries": entries,
    }
    write_json_new(manifest_path, payload)
    return payload
