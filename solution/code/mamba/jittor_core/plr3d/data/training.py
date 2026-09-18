"""Deterministic pure-Laplace PLR continuation dataset."""

from __future__ import annotations

import functools
import hashlib
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from jittor.dataset import Dataset

from ..io import PLRError, read_keys


def stable_item_seed(namespace: str, seed: int, epoch: int, index: int, key: str) -> int:
    payload = f"{namespace}:{seed}:{epoch}:{index}:{key}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def normalize_unit_sphere(array: np.ndarray) -> np.ndarray:
    point_max = array.max(axis=0, keepdims=True)
    point_min = array.min(axis=0, keepdims=True)
    center = (point_max + point_min) / 2.0
    centered = array.astype(np.float64) - center
    scale = float(np.linalg.norm(centered, axis=1).max())
    if not np.isfinite(scale) or scale <= 0.0:
        raise PLRError("invalid clean-cloud normalization scale")
    return (centered / scale).astype(np.float32)


def rotation_matrix(rng: np.random.RandomState) -> np.ndarray:
    angles = rng.uniform(-np.pi, np.pi, size=3)
    sx, sy, sz = np.sin(angles)
    cx, cy, cz = np.cos(angles)
    rotate_x = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, cx, sx], [0.0, -sx, cx]], dtype=np.float64
    )
    rotate_y = np.asarray(
        [[cy, 0.0, -sy], [0.0, 1.0, 0.0], [sy, 0.0, cy]], dtype=np.float64
    )
    rotate_z = np.asarray(
        [[cz, sz, 0.0], [-sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    return rotate_x @ rotate_y @ rotate_z


class PLRPatchDataset(Dataset):
    """On-the-fly specialist/tail patches used by PLR-001 and PLR-003."""

    def __init__(
        self,
        data_root: Path,
        keys: Sequence[str],
        samples_per_shape: int,
        patch_size: int,
        patch_ratio: float,
        sigma_by_category: Dict[str, Tuple[float, float]],
        gaussian_probability: float = 0.0,
        outlier_fraction: float = 0.0,
        outlier_scale: float = 3.0,
        scale_min: float = 1.0,
        scale_max: float = 1.0,
        seed: int = 20262731,
        seed_namespace: str = "plr003-tail-item",
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.keys = list(keys)
        self.samples_per_shape = int(samples_per_shape)
        self.patch_size = int(patch_size)
        self.patch_ratio = float(patch_ratio)
        self.sigma_by_category = dict(sigma_by_category)
        self.gaussian_probability = float(gaussian_probability)
        self.outlier_fraction = float(outlier_fraction)
        self.outlier_scale = float(outlier_scale)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.seed = int(seed)
        self.seed_namespace = str(seed_namespace)
        self.epoch = 0

        if not self.keys or self.samples_per_shape <= 0:
            raise PLRError("keys and samples_per_shape must be positive")
        if self.patch_size <= 0 or self.patch_ratio < 1.0:
            raise PLRError("invalid patch_size/patch_ratio")
        categories = {key.split("/")[1] for key in self.keys}
        if set(self.sigma_by_category) != categories:
            raise PLRError("sigma map categories must exactly match the training keys")
        if not 0.0 <= self.gaussian_probability <= 1.0:
            raise PLRError("gaussian_probability must be in [0,1]")
        if not 0.0 <= self.outlier_fraction <= 1.0:
            raise PLRError("outlier_fraction must be in [0,1]")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.keys) * self.samples_per_shape

    @functools.lru_cache(maxsize=64)
    def _load_clean(self, key: str) -> np.ndarray:
        path = self.data_root.joinpath(*key.split("/"), "clean.npy")
        array = np.load(path, allow_pickle=False)
        if array.dtype != np.float32 or array.ndim != 2 or array.shape[1] != 3:
            raise PLRError(f"invalid clean array {path}: {array.dtype} {array.shape}")
        if not np.isfinite(array).all():
            raise PLRError(f"non-finite clean array: {path}")
        return normalize_unit_sphere(array)

    def __getitem__(self, index: int):
        key = self.keys[index % len(self.keys)]
        rng = np.random.RandomState(
            stable_item_seed(self.seed_namespace, self.seed, self.epoch, index, key)
        )
        clean = self._load_clean(key).astype(np.float64, copy=True)
        clean = clean @ rotation_matrix(rng)
        clean *= rng.uniform(self.scale_min, self.scale_max)

        category = key.split("/")[1]
        sigma_min, sigma_max = self.sigma_by_category[category]
        sigma = float(rng.uniform(sigma_min, sigma_max))
        gaussian = bool(rng.rand() < self.gaussian_probability)
        if gaussian:
            noise = rng.normal(0.0, sigma, size=clean.shape)
            domain = 0
        else:
            noise = rng.laplace(0.0, sigma, size=clean.shape)
            outlier_mask = rng.rand(clean.shape[0]) < self.outlier_fraction
            noise[outlier_mask] *= self.outlier_scale
            domain = 1
        noisy = clean + noise

        seed_index = int(rng.randint(0, clean.shape[0]))
        seed_point = noisy[seed_index]
        noisy_distance = np.sum((noisy - seed_point) ** 2, axis=1)
        noisy_index = np.argpartition(noisy_distance, self.patch_size - 1)[: self.patch_size]
        noisy_index = noisy_index[
            np.argsort(noisy_distance[noisy_index], kind="stable")
        ]

        clean_count = int(round(self.patch_size * self.patch_ratio))
        clean_distance = np.sum((clean - seed_point) ** 2, axis=1)
        clean_index = np.argpartition(clean_distance, clean_count - 1)[:clean_count]
        clean_index = clean_index[
            np.argsort(clean_distance[clean_index], kind="stable")
        ]
        return {
            "domain": np.asarray(domain, dtype=np.int64),
            "pcl_clean": clean[clean_index].astype(np.float32),
            "pcl_noisy": noisy[noisy_index].astype(np.float32),
            "pcl_seeds": seed_point.astype(np.float32),
            "pcl_std": np.asarray(sigma, dtype=np.float32),
        }

    def collate_batch(self, batch):
        keys = batch[0].keys()
        return {key: np.stack([item[key] for item in batch], axis=0) for key in keys}
