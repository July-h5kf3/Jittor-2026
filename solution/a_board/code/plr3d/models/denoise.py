"""Jittor-native four-stage 3DMambaIPF denoiser and patch inference."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import jittor as jt
import numpy as np
from jittor import nn

from ..ops.geometry import farthest_point_sampling, knn_points
from .feature import FeatureExtraction


class DenoiseNet(nn.Module):
    def __init__(
        self,
        frame_knn: int = 32,
        num_modules: int = 4,
        noise_decay: float = 4.0,
    ) -> None:
        super().__init__()
        self.frame_knn = int(frame_knn)
        self.num_modules = int(num_modules)
        self.noise_decay = float(noise_decay)
        if self.num_modules <= 0:
            raise ValueError("num_modules must be positive")
        self.feature_nets = nn.ModuleList(
            [
                FeatureExtraction(
                    k=self.frame_knn,
                    input_dim=3,
                    z_dim=0,
                    embedding_dim=512,
                    output_dim=3,
                )
                for _ in range(self.num_modules)
            ]
        )
        self.last_patch_stitching_fallback_count = 0
        self.last_patch_stitching_fallback_indices = []

    def execute(self, points: jt.Var) -> jt.Var:
        return self.denoise_langevin_dynamics(points, self.num_modules)[0]

    def denoise_langevin_dynamics(
        self,
        noisy_points: jt.Var,
        num_modules_to_use: Optional[int] = None,
    ) -> Tuple[jt.Var, None]:
        if noisy_points.ndim != 3 or noisy_points.shape[2] != 3:
            raise ValueError("denoise_langevin_dynamics expects (B,N,3)")
        module_count = self.num_modules if num_modules_to_use is None else int(num_modules_to_use)
        if module_count <= 0 or module_count > self.num_modules:
            raise ValueError(f"module count must be in [1,{self.num_modules}]")
        current = noisy_points
        projection = None
        for index in range(module_count):
            displacement, projection = self.feature_nets[index](current, projection)
            current = current + displacement
        return current, None

    @jt.no_grad()
    def patch_based_denoise(
        self,
        noisy_points: jt.Var,
        patch_size: int = 2000,
        seed_k: int = 6,
        seed_k_alpha: int = 20,
        num_modules_to_use: Optional[int] = None,
    ) -> jt.Var:
        """Denoise one cloud and stitch exactly one output per input index."""
        if noisy_points.ndim != 2 or noisy_points.shape[1] != 3:
            raise ValueError("patch_based_denoise expects (N,3)")
        point_count = int(noisy_points.shape[0])
        if patch_size <= 0 or patch_size > point_count:
            raise ValueError("invalid patch_size")
        patch_count = int(seed_k * point_count / patch_size)
        if patch_count <= 0:
            raise ValueError("seed_k produces no patches")
        patch_step = int(point_count / (seed_k_alpha * patch_size))
        if patch_step <= 0:
            raise ValueError("seed_k_alpha is too large for this cloud/patch size")

        cloud = noisy_points.unsqueeze(0)
        seed_points, _ = farthest_point_sampling(cloud, patch_count)
        patch_distances, point_indices, patches = knn_points(
            seed_points,
            cloud,
            patch_size,
            return_nn=True,
        )
        if patches is None:
            raise RuntimeError("KNN did not return patch points")
        patches = patches[0]
        patch_distances = patch_distances[0]
        point_indices = point_indices[0].int32()
        seed_centers = seed_points[0].unsqueeze(1).broadcast(
            (patch_count, patch_size, 3)
        )
        patches = patches - seed_centers
        radius = patch_distances[:, -1:].broadcast(patch_distances.shape)
        patch_distances = patch_distances / jt.maximum(radius, 1e-20)

        all_distances = jt.ones((patch_count, point_count), dtype="float32") * float("inf")
        positions = jt.ones((patch_count, point_count), dtype="int32") * -1
        within_patch = jt.arange(patch_size).int32()
        for patch_index in range(patch_count):
            indices = point_indices[patch_index]
            all_distances[patch_index, indices] = patch_distances[patch_index]
            positions[patch_index, indices] = within_patch
        best_patch = jt.argmin(all_distances, dim=0)[0].int32()

        denoised_patches = []
        for start in range(0, patch_count, patch_step):
            current = patches[start : start + patch_step]
            denoised, _ = self.denoise_langevin_dynamics(
                current,
                num_modules_to_use=num_modules_to_use,
            )
            denoised_patches.append(denoised)
        denoised_patches = jt.concat(denoised_patches, dim=0) + seed_centers

        point_ids = jt.arange(point_count).int32()
        best_position = positions[best_patch, point_ids]
        covered = best_position >= 0
        safe_position = jt.maximum(best_position, 0).int32()
        selected = denoised_patches[best_patch, safe_position]
        original = cloud[0]
        output = jt.where(covered.unsqueeze(-1), selected, original)

        covered_numpy = covered.numpy().astype(bool, copy=False)
        fallback_indices = np.flatnonzero(~covered_numpy).astype(np.int64).tolist()
        self.last_patch_stitching_fallback_count = len(fallback_indices)
        self.last_patch_stitching_fallback_indices = fallback_indices
        if tuple(output.shape) != tuple(noisy_points.shape):
            raise RuntimeError(
                f"patch stitching changed shape {tuple(noisy_points.shape)} -> {tuple(output.shape)}"
            )
        return output

    def architecture_metadata(self) -> Dict[str, object]:
        return {
            "frame_knn": self.frame_knn,
            "num_modules": self.num_modules,
            "noise_decay": self.noise_decay,
            "embedding_dim": 512,
            "mamba_version_compatibility": "1.1.3.post1",
        }
