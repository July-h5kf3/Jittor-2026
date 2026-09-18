#!/usr/bin/env python3
"""Jittor training entry for the StraightPCF velocity module.

The implementation uses Jittor, NumPy and the Python standard library.  Mesh
sampling, Laplace noise and patch construction follow the same B-board VM
recipe as the submitted checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import jittor as jt
import numpy as np
from jittor import optim

from model_vm import (
    DECODER_HIDDEN_DIM,
    DSM_SIGMA,
    EMBEDDING_DIM,
    FRAME_KNN,
    TRAIN_POINTS,
    VelocityModule,
)

POINT_COUNT = 50000
PATCH_COUNT = 4
PATCH_SIZE = 1000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init-weights", type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--base-lr", type=float, default=5e-5)
    parser.add_argument("--min-lr", type=float, default=5e-7)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--vertex-samples", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=0)
    return parser.parse_args()


def read_keys(path: Path) -> list[str]:
    keys = [
        line.strip().replace("\\", "/").strip("/")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("train list is empty or duplicated: %s" % path)
    return keys


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    cache = Path(str(path) + ".cache.npz")
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as payload:
            vertices = np.asarray(payload["vertices"], dtype=np.float64)
            faces = np.asarray(payload["faces"], dtype=np.int64)
            return vertices, faces
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
    vertices = np.asarray(vertex_rows, dtype=np.float64)
    faces = np.asarray(face_rows, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not faces.size:
        raise ValueError("invalid mesh: %s" % path)
    return vertices, faces


def sample_cloud(vertices: np.ndarray, faces: np.ndarray, vertex_samples: int, rng: np.random.RandomState) -> np.ndarray:
    take = min(int(vertex_samples), vertices.shape[0], POINT_COUNT)
    selected = rng.choice(vertices.shape[0], size=take, replace=False)
    remaining = POINT_COUNT - take
    edge0 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    edge1 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    areas = np.linalg.norm(np.cross(edge0, edge1), axis=1)
    total = float(areas.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("mesh has no positive-area face")
    face_index = rng.choice(len(faces), size=remaining, replace=True, p=areas / total)
    chosen = vertices[faces[face_index]]
    uv = rng.rand(remaining, 2)
    reflected = (uv[:, 0] + uv[:, 1]) > 1.0
    uv[reflected] = 1.0 - uv[reflected]
    surface = (
        chosen[:, 0]
        + uv[:, 0:1] * (chosen[:, 1] - chosen[:, 0])
        + uv[:, 1:2] * (chosen[:, 2] - chosen[:, 0])
    )
    clean = np.concatenate([vertices[selected], surface], axis=0)
    clean = clean[rng.permutation(POINT_COUNT)]
    center = 0.5 * (clean.max(axis=0, keepdims=True) + clean.min(axis=0, keepdims=True))
    clean = clean - center
    radius = float(np.linalg.norm(clean, axis=1).max())
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("invalid sampled radius")
    return np.ascontiguousarray(clean / radius, dtype=np.float32)


def rotation_matrix(rng: np.random.RandomState) -> np.ndarray:
    matrix = np.eye(3, dtype=np.float32)
    for axis in range(3):
        angle = float(rng.uniform(-math.pi, math.pi))
        cosine, sine = math.cos(angle), math.sin(angle)
        if axis == 0:
            current = np.asarray([[1, 0, 0], [0, cosine, sine], [0, -sine, cosine]], dtype=np.float32)
        elif axis == 1:
            current = np.asarray([[cosine, 0, -sine], [0, 1, 0], [sine, 0, cosine]], dtype=np.float32)
        else:
            current = np.asarray([[cosine, sine, 0], [-sine, cosine, 0], [0, 0, 1]], dtype=np.float32)
        matrix = matrix @ current
    return matrix


def nearest_indices(points: np.ndarray, query: np.ndarray, count: int) -> np.ndarray:
    delta = points - query
    distance = np.einsum("ij,ij->i", delta, delta)
    return np.argpartition(distance, count - 1)[:count]


def farthest_indices(points: np.ndarray, count: int, rng: np.random.RandomState) -> np.ndarray:
    indices = np.empty(count, dtype=np.int64)
    indices[0] = int(rng.randint(len(points)))
    nearest = np.full(len(points), np.inf, dtype=np.float64)
    for slot in range(count):
        if slot:
            indices[slot] = int(np.argmax(nearest))
        delta = points - points[indices[slot]]
        nearest = np.minimum(nearest, np.einsum("ij,ij->i", delta, delta))
    return indices


def build_mesh_batch(mesh: Path, vertex_samples: int, rng: np.random.RandomState) -> dict[str, np.ndarray]:
    vertices, faces = load_mesh(mesh)
    clean = sample_cloud(vertices, faces, vertex_samples, rng)
    clean = clean @ rotation_matrix(rng)
    noisy = clean + rng.laplace(0.0, 0.01, clean.shape).astype(np.float32)
    centers = farthest_indices(noisy, PATCH_COUNT, rng)
    noisy_rows = []
    clean_rows = []
    interpolated_rows = []
    for center_id in centers:
        noisy_ids = nearest_indices(noisy, noisy[center_id], PATCH_SIZE)
        t = float(rng.uniform(1e-8, 1.0))
        noisy_l2 = noisy[noisy_ids]
        clean_l2 = clean[noisy_ids]
        interpolated = np.float32(t) * clean_l2 + np.float32(1.0 - t) * noisy_l2
        seed_t = np.float32(t) * clean[center_id] + np.float32(1.0 - t) * noisy[center_id]
        noisy_rows.append(noisy_l2 - seed_t)
        clean_rows.append(clean_l2 - seed_t)
        interpolated_rows.append(interpolated - seed_t)
    return {
        "noisy_l2": np.ascontiguousarray(noisy_rows, dtype=np.float32),
        "clean": np.ascontiguousarray(clean_rows, dtype=np.float32),
        "interpolated": np.ascontiguousarray(interpolated_rows, dtype=np.float32),
    }


def learning_rate(step: int, total_steps: int, base_lr: float, min_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * float(step + 1) / float(max(1, warmup))
    progress = min(1.0, float(step - warmup) / float(max(1, total_steps - warmup - 1)))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def collect_arrays(model: VelocityModule) -> dict[str, np.ndarray]:
    arrays = {
        name: np.ascontiguousarray(param.numpy())
        for name, param in model.state_dict().items()
    }
    arrays["decoder.bn_1_out.running_mean"] = np.ascontiguousarray(model.decoder.bn_1_out.running_mean.numpy())
    arrays["decoder.bn_1_out.running_var"] = np.ascontiguousarray(model.decoder.bn_1_out.running_var.numpy())
    arrays["decoder.bn_2_out.running_mean"] = np.ascontiguousarray(model.decoder.bn_2_out.running_mean.numpy())
    arrays["decoder.bn_2_out.running_var"] = np.ascontiguousarray(model.decoder.bn_2_out.running_var.numpy())
    return arrays


def save_npz(model: VelocityModule, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **collect_arrays(model))


def mesh_path(root: Path, key: str) -> Path:
    options = [
        root.joinpath(*key.split("/"), "models", "model_normalized.obj"),
        root.joinpath("shapenet", *key.split("/"), "models", "model_normalized.obj"),
    ]
    for path in options:
        if path.is_file():
            return path
    raise FileNotFoundError("%s / %s" % (root, key))


def main() -> int:
    args = parse_args()
    jt.flags.use_cuda = 1
    random.seed(args.seed)
    np.random.seed(args.seed & 0xFFFFFFFF)
    jt.set_global_seed(args.seed)

    keys = read_keys(args.train_list)
    model = VelocityModule()
    if args.init_weights is not None:
        from infer_vm import load_weights

        load_weights(model, args.init_weights)
    model.train()
    optimizer = optim.Adam(model.parameters(), args.base_lr, weight_decay=0.0)

    steps_per_epoch = len(keys)
    total_steps = steps_per_epoch * args.epochs
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        "framework": "Jittor",
        "model": "StraightPCF VM",
        "variant": "fixed_v8192",
        "epochs": args.epochs,
        "base_lr": args.base_lr,
        "min_lr": args.min_lr,
        "warmup_steps": args.warmup_steps,
        "grad_clip": args.grad_clip,
        "seed": args.seed,
        "optimizer": "Adam",
        "laplace_scale": 0.01,
        "vertex_samples": args.vertex_samples,
        "patch_size": PATCH_SIZE,
        "patches_per_mesh": PATCH_COUNT,
        "dsm_sigma": DSM_SIGMA,
        "train_points": TRAIN_POINTS,
        "frame_knn": FRAME_KNN,
        "embedding_dim": EMBEDDING_DIM,
        "decoder_hidden_dim": DECODER_HIDDEN_DIM,
        "train_count": len(keys),
        "steps_per_epoch": steps_per_epoch,
        "train_list_sha256": hashlib.sha256(args.train_list.read_bytes()).hexdigest(),
    }
    (args.output / "training_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    global_step = 0
    started = time.time()
    for epoch in range(args.epochs):
        order = np.random.RandomState(args.seed + epoch).permutation(len(keys))
        for local_step, key_index in enumerate(order):
            if args.max_steps and global_step >= args.max_steps:
                break
            key = keys[int(key_index)]
            rng = np.random.RandomState((args.seed + epoch * 1000003 + int(key_index)) & 0xFFFFFFFF)
            batch = build_mesh_batch(mesh_path(args.data_root, key), args.vertex_samples, rng)
            interpolated = jt.array(batch["interpolated"])
            noisy_l2 = jt.array(batch["noisy_l2"])
            clean = jt.array(batch["clean"])
            point_ids = jt.array(rng.permutation(PATCH_SIZE)[:TRAIN_POINTS].astype(np.int32))
            loss = model(interpolated, noisy_l2, clean, point_ids)
            current_lr = learning_rate(
                global_step, total_steps, args.base_lr, args.min_lr, args.warmup_steps
            )
            optimizer.lr = current_lr
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.clip_grad_norm(args.grad_clip, 2)
            optimizer.step()
            global_step += 1
            if global_step % args.log_every == 0 or local_step + 1 == len(order):
                print(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "step": global_step,
                            "loss": float(loss.item()),
                            "lr": current_lr,
                            "elapsed": time.time() - started,
                            "key": key,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        checkpoint = args.output / ("fixed_v8192_epoch%02d.npz" % (epoch + 1))
        save_npz(model, checkpoint)
        print(json.dumps({"event": "save", "checkpoint": str(checkpoint)}, ensure_ascii=False), flush=True)
        if args.max_steps and global_step >= args.max_steps:
            break
    save_npz(model, args.output / "fixed_v8192_epoch10.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
