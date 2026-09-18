"""Deterministic overlap stitching shared by canonical and streaming Jittor inference."""

from __future__ import annotations

import numpy as np


def merge_patch_predictions(
    predictions: np.ndarray,
    positions: np.ndarray,
    normalized_distances: np.ndarray,
    original: np.ndarray,
    strategy: str = "softmax010",
    temperature: float = 0.10,
) -> tuple[np.ndarray, list[int]]:
    denoised = np.asarray(predictions, dtype=np.float32)
    local = np.asarray(positions, dtype=np.int32)
    distances = np.asarray(normalized_distances, dtype=np.float32)
    source = np.asarray(original, dtype=np.float32)
    if denoised.ndim != 3 or denoised.shape[2] != 3:
        raise ValueError("predictions must have shape (patch,point,3)")
    if local.ndim != 2 or local.shape != distances.shape:
        raise ValueError("positions/distances shape mismatch")
    if local.shape[0] != denoised.shape[0] or source.shape != (local.shape[1], 3):
        raise ValueError("patch/point dimensions do not agree")
    valid = local >= 0
    if np.any(local[valid] >= denoised.shape[1]):
        raise ValueError("invalid patch-local index")
    if np.any(~np.isfinite(distances[valid])) or np.any(np.isfinite(distances[~valid])):
        raise ValueError("distance coverage contract failed")
    covered = valid.any(axis=0)
    output = source.copy()
    if strategy == "best_patch":
        point_ids = np.arange(source.shape[0], dtype=np.int64)
        best_patch = np.argmin(distances, axis=0)
        best_local = local[best_patch, point_ids]
        output[covered] = denoised[best_patch[covered], best_local[covered]]
    elif strategy == "softmax010":
        if abs(float(temperature) - 0.10) > 1e-12:
            raise ValueError("softmax010 requires temperature 0.10")
        minimum = np.min(distances, axis=0)
        numerator = np.zeros_like(source, dtype=np.float32)
        denominator = np.zeros(source.shape[0], dtype=np.float32)
        for patch_id in range(local.shape[0]):
            point_ids = np.flatnonzero(valid[patch_id])
            if point_ids.size == 0:
                continue
            local_ids = local[patch_id, point_ids]
            weight = np.exp(
                -(distances[patch_id, point_ids] - minimum[point_ids])
                / np.float32(temperature)
            ).astype(np.float32)
            numerator[point_ids] += denoised[patch_id, local_ids] * weight[:, None]
            denominator[point_ids] += weight
        output[covered] = numerator[covered] / denominator[covered, None]
    else:
        raise ValueError("unknown merge strategy: %s" % strategy)
    if output.dtype != np.float32 or output.shape != source.shape or not np.isfinite(output).all():
        raise RuntimeError("patch merge violated the output contract")
    fallback = np.flatnonzero(~covered).astype(np.int64).tolist()
    return np.ascontiguousarray(output), fallback
