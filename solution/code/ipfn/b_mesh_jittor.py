"""NumPy mesh sampler for end-to-end Jittor IterativePFN training."""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


DEFAULT_POINT_COUNT = 50000
DEFAULT_VERTEX_SAMPLES = 8192
DEFAULT_PATCH_SIZE = 1000
DEFAULT_PATCH_RATIO = 1.2
DEFAULT_PATCHES_PER_MESH = 4
DEFAULT_LAPLACE_SCALE = 0.010
DEFAULT_SCALE_JITTER_PROBABILITY = 0.75
DEFAULT_SCALE_MIN = 0.95
DEFAULT_SCALE_MAX = 1.05


PROFILE = [
    ("laplace", 0.60, 0.008, 0.012),
    ("laplace", 0.15, 0.005, 0.020),
    ("gaussian", 0.15, 0.008, 0.012),
    ("anisotropic_gaussian", 0.10, 0.008, 0.012),
]


class _Mesh:
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces


def load_mesh(path):
    path = Path(path)
    cache = Path(str(path) + ".cache.npz")
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as payload:
            vertices = np.asarray(payload["vertices"], dtype=np.float32)
            faces = np.asarray(payload["faces"], dtype=np.int64)
            return _Mesh(vertices, faces)
    vertex_rows = []
    face_rows = []
    for raw in path.read_text(encoding="utf-8", errors="strict").splitlines():
        line = raw.strip()
        if line.startswith("v "):
            vertex_rows.append([float(value) for value in line.split()[1:4]])
        elif line.startswith("f "):
            indices = []
            for token in line.split()[1:]:
                raw_index = int(token.split("/", 1)[0])
                indices.append(raw_index - 1 if raw_index > 0 else len(vertex_rows) + raw_index)
            for offset in range(1, len(indices) - 1):
                face_rows.append([indices[0], indices[offset], indices[offset + 1]])
    vertices = np.asarray(vertex_rows, dtype=np.float32)
    faces = np.asarray(face_rows, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not faces.size:
        raise ValueError("invalid mesh: %s" % path)
    return _Mesh(vertices, faces)


def sample_surface(mesh, count, vertex_samples=DEFAULT_VERTEX_SAMPLES):
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    take = min(int(vertex_samples), int(count), len(vertices))
    vertex_part = vertices[np.random.choice(len(vertices), take, replace=False)] if take else np.empty((0, 3), np.float32)
    remaining = int(count) - take
    if remaining > 0:
        triangles = vertices[faces]
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        areas = np.linalg.norm(cross, axis=1).astype(np.float64)
        total = float(areas.sum())
        if not math.isfinite(total) or total <= 0:
            raise ValueError("mesh has no positive-area triangle")
        face_ids = np.random.choice(len(faces), size=remaining, replace=True, p=areas / total)
        chosen = triangles[face_ids]
        uv = np.random.random((remaining, 2)).astype(np.float32)
        reflected = (uv[:, 0] + uv[:, 1]) > 1.0
        uv[reflected] = 1.0 - uv[reflected]
        surface_part = (
            chosen[:, 0]
            + uv[:, 0:1] * (chosen[:, 1] - chosen[:, 0])
            + uv[:, 1:2] * (chosen[:, 2] - chosen[:, 0])
        ).astype(np.float32)
    else:
        surface_part = np.empty((0, 3), np.float32)
    points = np.concatenate([vertex_part, surface_part], axis=0)
    np.random.shuffle(points)
    center = 0.5 * (points.max(0, keepdims=True) + points.min(0, keepdims=True))
    points -= center
    points /= max(float(np.linalg.norm(points, axis=1).max()), 1e-12)
    return np.ascontiguousarray(points, dtype=np.float32)


def draw_noise(shape):
    bucket, cumulative = random.random(), 0.0
    for kind, weight, low, high in PROFILE:
        cumulative += weight
        if bucket <= cumulative + 1e-12:
            break
    scale = random.uniform(low, high)
    axis = np.ones((1, 3), np.float32)
    if kind == "laplace":
        noise = np.random.laplace(0, scale, shape)
    elif kind == "gaussian":
        noise = np.random.normal(0, scale, shape)
    else:
        axis = np.random.uniform(0.5, 1.8, (1, 3)).astype(np.float32)
        axis /= np.sqrt(np.mean(axis * axis))
        noise = np.random.normal(0, scale, shape) * axis
    return np.asarray(noise, np.float32), np.float32(scale), kind, axis


def rotation_matrix():
    matrix = np.eye(3, dtype=np.float32)
    for axis in range(3):
        angle = random.uniform(-math.pi, math.pi)
        c, s = math.cos(angle), math.sin(angle)
        current = (
            np.array([[1, 0, 0], [0, c, s], [0, -s, c]], np.float32) if axis == 0 else
            np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]], np.float32) if axis == 1 else
            np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], np.float32)
        )
        matrix = matrix @ current
    return matrix


def draw_mesh_scale(
    probability=DEFAULT_SCALE_JITTER_PROBABILITY,
    low=DEFAULT_SCALE_MIN,
    high=DEFAULT_SCALE_MAX,
):
    """Draw exactly one isotropic scale decision for a complete mesh sample."""
    branch_draw = random.random()
    if branch_draw < float(probability):
        return np.float32(random.uniform(float(low), float(high))), True, branch_draw
    return np.float32(1.0), False, branch_draw


def farthest_indices(points, count):
    indices = np.empty(count, np.int64)
    indices[0] = random.randrange(len(points))
    nearest = np.full(len(points), np.inf)
    for slot in range(count):
        if slot:
            indices[slot] = int(np.argmax(nearest))
        delta = points - points[indices[slot]]
        nearest = np.minimum(nearest, np.einsum("ij,ij->i", delta, delta))
    return indices


class BMeshSampler:
    def __init__(
        self,
        mesh_root,
        datalist,
        patch_size=DEFAULT_PATCH_SIZE,
        patch_ratio=DEFAULT_PATCH_RATIO,
        patches_per_mesh=DEFAULT_PATCHES_PER_MESH,
        full_counts=(DEFAULT_POINT_COUNT,),
        patch_mode="patch",
        vertex_samples=DEFAULT_VERTEX_SAMPLES,
        laplace_scale=DEFAULT_LAPLACE_SCALE,
        scale_jitter_probability=DEFAULT_SCALE_JITTER_PROBABILITY,
        scale_min=DEFAULT_SCALE_MIN,
        scale_max=DEFAULT_SCALE_MAX,
    ):
        rows = [x.strip() for x in Path(datalist).read_text(encoding="utf-8").splitlines() if x.strip()]
        root = Path(mesh_root)
        self.paths = [root / row / "models/model_normalized.obj" for row in rows]
        self.patch_size = int(patch_size)
        self.clean_size = int(round(patch_size * patch_ratio))
        self.patches_per_mesh = int(patches_per_mesh)
        self.full_counts = tuple(full_counts)
        self.patch_mode = str(patch_mode)
        self.vertex_samples = int(vertex_samples)
        self.laplace_scale = float(laplace_scale)
        self.scale_jitter_probability = float(scale_jitter_probability)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        if self.patch_mode not in ("patch", "full"):
            raise ValueError("patch_mode must be 'patch' or 'full'")
        if self.patch_mode == "full" and self.full_counts != (DEFAULT_POINT_COUNT,):
            raise ValueError("full mode requires exactly 50000 points")
        if not (
            self.vertex_samples == DEFAULT_VERTEX_SAMPLES
            and self.laplace_scale == DEFAULT_LAPLACE_SCALE
            and 0.0 <= self.scale_jitter_probability <= 1.0
            and 0.0 < self.scale_min <= self.scale_max
        ):
            raise ValueError("invalid locked augmentation configuration")

    def __len__(self):
        return len(self.paths)

    def sample(self, index):
        clean = sample_surface(
            load_mesh(self.paths[int(index) % len(self.paths)]),
            random.choice(self.full_counts),
            vertex_samples=self.vertex_samples,
        )
        rotation = rotation_matrix()
        augment_scale, jitter_applied, branch_draw = draw_mesh_scale(
            self.scale_jitter_probability, self.scale_min, self.scale_max
        )
        # The complete mesh shares one rotation and one isotropic scale.  Noise
        # is drawn afterwards so final coordinates retain independent,
        # axis-aligned Laplace(0, 0.010) variables.
        clean = (clean @ rotation) * augment_scale
        noise = np.random.laplace(0.0, self.laplace_scale, clean.shape).astype(np.float32)
        noisy = clean + noise
        scale = np.float32(self.laplace_scale)
        kind = "laplace"
        transform = np.eye(3, dtype=np.float32)
        if self.patch_mode == "full":
            return {
                "noisy": np.ascontiguousarray(noisy[None], np.float32),
                "clean": np.ascontiguousarray(clean[None], np.float32),
                "seeds": np.zeros((1, 1, 3), dtype=np.float32),
                "scale": np.asarray([scale], dtype=np.float32),
                "kind": [kind],
                "transform": transform[None],
                "mesh_scale": float(augment_scale),
                "scale_jitter_applied": bool(jitter_applied),
                "scale_branch_draw": float(branch_draw),
                "patch_mode": self.patch_mode,
            }
        noisy_tree, clean_tree = cKDTree(noisy), cKDTree(clean)
        noisy_patches, clean_patches, seeds = [], [], []
        for seed_index in farthest_indices(noisy, self.patches_per_mesh):
            seed = noisy[seed_index:seed_index + 1]
            noisy_idx = noisy_tree.query(seed[0], k=self.patch_size, workers=1)[1]
            clean_idx = clean_tree.query(seed[0], k=self.clean_size, workers=1)[1]
            noisy_patches.append(noisy[noisy_idx])
            clean_patches.append(clean[clean_idx])
            seeds.append(seed)
        return {
            "noisy": np.ascontiguousarray(noisy_patches, np.float32),
            "clean": np.ascontiguousarray(clean_patches, np.float32),
            "seeds": np.ascontiguousarray(seeds, np.float32),
            "scale": np.full((self.patches_per_mesh,), scale, np.float32),
            "kind": [kind] * self.patches_per_mesh,
            "transform": np.repeat(transform[None], self.patches_per_mesh, 0).astype(np.float32),
            "mesh_scale": float(augment_scale),
            "scale_jitter_applied": bool(jitter_applied),
            "scale_branch_draw": float(branch_draw),
            "patch_mode": self.patch_mode,
        }
