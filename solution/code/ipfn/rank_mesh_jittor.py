"""Online rank-candidate mesh sampler matching the fixed validation recipe."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from b_mesh_jittor import (
    DEFAULT_LAPLACE_SCALE,
    DEFAULT_PATCHES_PER_MESH,
    DEFAULT_PATCH_RATIO,
    DEFAULT_PATCH_SIZE,
    DEFAULT_POINT_COUNT,
    DEFAULT_SCALE_JITTER_PROBABILITY,
    DEFAULT_SCALE_MAX,
    DEFAULT_SCALE_MIN,
    DEFAULT_VERTEX_SAMPLES,
    BMeshSampler,
    farthest_indices,
    load_mesh,
    rotation_matrix,
    sample_surface,
)


class RankMeshSampler(BMeshSampler):
    """BMeshSampler-compatible loader whose datalist contains complete keys."""

    def __init__(self, mesh_root, datalist, **kwargs):
        rows = [x.strip() for x in Path(datalist).read_text(encoding="utf-8").splitlines() if x.strip()]
        self.paths = [Path(mesh_root) / row / "models/model_normalized.obj" for row in rows]
        missing = [str(path) for path in self.paths if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing %d meshes; first=%s" % (len(missing), missing[0]))
        self.patch_size = int(kwargs.get("patch_size", DEFAULT_PATCH_SIZE))
        self.clean_size = int(round(self.patch_size * float(kwargs.get("patch_ratio", DEFAULT_PATCH_RATIO))))
        self.patches_per_mesh = int(kwargs.get("patches_per_mesh", DEFAULT_PATCHES_PER_MESH))
        self.patch_mode = str(kwargs.get("patch_mode", "patch"))
        self.vertex_samples = int(kwargs.get("vertex_samples", DEFAULT_VERTEX_SAMPLES))
        self.laplace_scale = float(kwargs.get("laplace_scale", DEFAULT_LAPLACE_SCALE))
        self.scale_jitter_probability = float(
            kwargs.get("scale_jitter_probability", DEFAULT_SCALE_JITTER_PROBABILITY)
        )
        self.scale_min = float(kwargs.get("scale_min", DEFAULT_SCALE_MIN))
        self.scale_max = float(kwargs.get("scale_max", DEFAULT_SCALE_MAX))
        if self.patch_mode not in ("patch", "full"):
            raise ValueError("patch_mode must be 'patch' or 'full'")
        # Validation always uses 50k points and axis-independent Laplace(0.010).
        self.full_counts = (DEFAULT_POINT_COUNT,)

    def sample(self, index):
        return BMeshSampler.sample(self, index)
