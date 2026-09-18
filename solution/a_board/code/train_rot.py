#!/usr/bin/env python3
"""Train Jittor StraightPCF VM -> CVM -> SPCF from original ShapeNet meshes.

The implementation intentionally uses only Jittor, NumPy and the Python standard
library.  It replaces the research repository's OmegaConf/SciPy/trimesh runner
while preserving its mesh sampling, noise, patch construction and model losses.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

import jittor as jt
import numpy as np
from jittor import optim
from jittor.dataset import Dataset

from plr3d.io import PLRError, sha256_file

PACKAGE_ROOT = Path(__file__).resolve().parent
ROT_ROOT = PACKAGE_ROOT / "rot_jittor"
if str(ROT_ROOT) not in sys.path:
    sys.path.insert(0, str(ROT_ROOT))

from src.model.straightpcf import StraightPCFModule  # noqa: E402

STAGES = ("vm", "cvm", "spcf")
BASE_CONFIG = {
    "frame_knn": 32,
    "num_train_points": 128,
    "feat_embedding_dim": 256,
    "decoder_hidden_dim": 64,
    "dsm_sigma": 0.01,
    "num_modules": 4,
    "tot_its": 2,
    "decoder_type": "graph",
    "distance_estimation": True,
    "film": True,
    "multiscale": False,
    "distance_multiscale": True,
    "predict_alpha": 1.05,
    "predict_passes": 1,
    "predict_tta": 0,
    "predict_fusion": False,
}


def read_keys(path: Path) -> list[str]:
    keys = [line.strip().replace("\\", "/").strip("/") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not keys or len(keys) != len(set(keys)):
        raise PLRError(f"ROT key list is empty or duplicated: {path}")
    if any(len(key.split("/")) != 3 or not key.startswith("shapenet/") for key in keys):
        raise PLRError(f"invalid ROT key list: {path}")
    return keys


def stable_seed(namespace: str, seed: int, epoch: int, index: int, key: str) -> int:
    payload = f"{namespace}:{seed}:{epoch}:{index}:{key}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


@functools.lru_cache(maxsize=32)
def load_mesh(path_text: str) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path_text)
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
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise PLRError(f"invalid ROT mesh vertices: {path}")
    if faces.ndim != 2 or faces.shape[1] != 3 or not faces.size:
        raise PLRError(f"invalid ROT mesh faces: {path}")
    if faces.min() < 0 or faces.max() >= vertices.shape[0]:
        raise PLRError(f"ROT face index out of range: {path}")
    return vertices, faces


def sample_training_cloud(vertices: np.ndarray, faces: np.ndarray, count: int, vertex_count: int, rng: np.random.RandomState) -> np.ndarray:
    vertex_count = min(vertex_count, vertices.shape[0], count)
    selected_vertices = rng.permutation(vertices.shape[0])[:vertex_count]
    remaining = count - vertex_count
    edge0 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    edge1 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    cumulative = np.cumsum(np.linalg.norm(np.cross(edge0, edge1), axis=1), dtype=np.float64)
    if not np.isfinite(cumulative[-1]) or cumulative[-1] <= 0.0:
        raise PLRError("ROT mesh has no finite positive-area face")
    face_index = np.searchsorted(cumulative, rng.random_sample(remaining) * cumulative[-1])
    origins = vertices[faces[:, 0]][face_index]
    vectors = (vertices[faces[:, 1:]] - vertices[faces[:, 0]][:, None, :])[face_index]
    lengths = rng.random_sample((remaining, 2, 1))
    reflected = lengths.sum(axis=1).reshape(-1) > 1.0
    lengths[reflected] -= 1.0
    surface = origins + (vectors * np.abs(lengths)).sum(axis=1)
    return np.concatenate([vertices[selected_vertices], surface], axis=0)


def euler_zyx_matrix(x_degrees: float, y_degrees: float, z_degrees: float) -> np.ndarray:
    x, y, z = np.deg2rad([x_degrees, y_degrees, z_degrees])
    sx, sy, sz = np.sin([x, y, z])
    cx, cy, cz = np.cos([x, y, z])
    rx = np.asarray([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry = np.asarray([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz = np.asarray([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def build_patch(vertices: np.ndarray, faces: np.ndarray, rng: np.random.RandomState, *, surface_points: int, vertex_samples: int, patch_size: int, validation: bool) -> Dict[str, np.ndarray]:
    clean = sample_training_cloud(vertices, faces, surface_points, vertex_samples, rng)
    center = (clean.max(axis=0) + clean.min(axis=0)) / 2.0
    clean = clean - center
    radius = float(np.sqrt((clean * clean).sum(axis=1).max()))
    if not np.isfinite(radius) or radius <= 0.0:
        raise PLRError("invalid ROT normalization radius")
    clean = clean / radius
    sigma = 0.011 if validation else float(rng.uniform(0.008, 0.014))
    noise = rng.laplace(0.0, sigma, size=clean.shape)
    outlier_mask = rng.rand(clean.shape[0]) < 0.02
    noise[outlier_mask] *= 3.0
    noisy = clean + noise
    if not validation:
        if rng.rand() < 0.5:
            rotation = euler_zyx_matrix(*rng.uniform(-180.0, 180.0, size=3))
            clean = clean @ rotation.T
            noisy = noisy @ rotation.T
        if rng.rand() < 0.5:
            scale = rng.uniform(0.8, 1.2, size=(1, 3))
            clean = clean * scale
            noisy = noisy * scale
    seed_index = int(rng.permutation(noisy.shape[0])[0])
    delta = noisy - noisy[seed_index]
    distance = np.sum(delta * delta, axis=1)
    nearest = np.argpartition(distance, patch_size - 1)[:patch_size]
    nearest = nearest[np.argsort(distance[nearest], kind="stable")]
    pcl_noisy = noisy[nearest]
    pcl_clean = clean[nearest]
    t = float(rng.uniform(1e-8, 1.0))
    seed_t = t * clean[seed_index] + (1.0 - t) * noisy[seed_index]
    return {
        "pcl_noisy_L2": pcl_noisy[None].astype(np.float32),
        "pcl_clean": pcl_clean[None].astype(np.float32),
        "seed_points_t": seed_t.reshape(1, 1, 3).astype(np.float32),
        "original_time_step": np.asarray([t], dtype=np.float32),
    }


class ROTPatchDataset(Dataset):
    def __init__(self, mesh_root: Path, keys: Sequence[str], total: int, seed: int, namespace: str, surface_points: int, vertex_samples: int, patch_size: int, validation: bool):
        super().__init__()
        self.mesh_root = Path(mesh_root)
        self.keys = list(keys)
        self.total = int(total)
        self.seed = int(seed)
        self.namespace = namespace
        self.surface_points = int(surface_points)
        self.vertex_samples = int(vertex_samples)
        self.patch_size = int(patch_size)
        self.validation = bool(validation)
        self.epoch = 0

    def __len__(self) -> int:
        return self.total

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int):
        key = self.keys[index % len(self.keys)]
        mesh = self.mesh_root.joinpath(*key.split("/"), "models", "model_normalized.obj")
        if not mesh.is_file():
            raise FileNotFoundError(mesh)
        rng = np.random.RandomState(stable_seed(self.namespace, self.seed, self.epoch, index, key))
        vertices, faces = load_mesh(str(mesh))
        return build_patch(vertices, faces, rng, surface_points=self.surface_points,
                           vertex_samples=self.vertex_samples, patch_size=self.patch_size,
                           validation=self.validation)

    def collate_batch(self, batch):
        return {key: np.stack([item[key] for item in batch], axis=0) for key in batch[0]}


def as_jittor(batch):
    return {key: value if isinstance(value, jt.Var) else jt.array(value) for key, value in batch.items()}


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def save_checkpoint(path: Path, model: StraightPCFModule, metadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    jt.save({"state_dict": model.state_dict(), "metadata": metadata}, str(temporary))
    os.replace(temporary, path)


def load_checkpoint(path: Path, model: StraightPCFModule):
    payload = jt.load(str(path))
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    model.load_parameters(state)
    return payload.get("metadata", {}) if isinstance(payload, dict) else {}


def mean_loss(model, loader) -> float:
    model.eval()
    values = []
    with jt.no_grad():
        for batch in loader:
            loss = model.training_step(as_jittor(batch))["loss"]
            value = float(loss.item())
            if not math.isfinite(value):
                raise PLRError("non-finite ROT validation loss")
            values.append(value)
    return float(np.mean(values)) if values else float("inf")


def train_stage(stage: str, epochs: int, previous: Optional[Path], args, train_keys, validation_keys):
    stage_root = args.output_root / stage
    stage_root.mkdir(parents=True, exist_ok=True)
    latest = stage_root / f"{stage}-latest.pkl"
    best = stage_root / f"{stage}-best.pkl"
    state_path = stage_root / "stage_state.json"
    if state_path.is_file() and best.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("completed") and int(state.get("requested_epochs", -1)) == epochs:
            return best, state
    config = dict(BASE_CONFIG)
    config["stage"] = stage
    random.seed(args.seed + STAGES.index(stage))
    np.random.seed(args.seed + STAGES.index(stage))
    jt.set_global_seed(args.seed + STAGES.index(stage))
    model = StraightPCFModule(config, {})
    start_epoch = 0
    best_loss = float("inf")
    best_epoch = 0
    records = []
    if latest.is_file():
        if not args.resume:
            raise PLRError(f"partial ROT stage exists; pass --resume: {latest}")
        metadata = load_checkpoint(latest, model)
        start_epoch = int(metadata.get("epoch", 0))
        best_loss = float(metadata.get("best_validation_loss", float("inf")))
        best_epoch = int(metadata.get("best_epoch", 0))
        records = list(metadata.get("records", []))
    elif previous is not None:
        model.init_from_stage(str(previous))
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_set = ROTPatchDataset(args.mesh_root, train_keys, args.samples_per_epoch,
                                args.seed, f"rot-{stage}-train", args.surface_points,
                                args.vertex_samples, args.patch_size, False)
    train_set.set_attrs(batch_size=args.batch_size, total_len=len(train_set), shuffle=True,
                        num_workers=args.num_workers, drop_last=False)
    validation_set = ROTPatchDataset(args.mesh_root, validation_keys, min(args.validation_count, len(validation_keys)),
                                     args.seed + 1000, f"rot-{stage}-validation", args.surface_points,
                                     args.vertex_samples, args.patch_size, True)
    validation_set.set_attrs(batch_size=args.validation_batch_size, total_len=len(validation_set),
                             shuffle=False, num_workers=args.num_workers, drop_last=False)
    for epoch in range(start_epoch, epochs):
        train_set.set_epoch(epoch)
        model.train()
        started = time.time()
        total = 0.0
        steps = 0
        for batch in train_set:
            loss = model.training_step(as_jittor(batch))["loss"]
            if not bool(jt.isfinite(loss).all().item()):
                raise PLRError(f"non-finite ROT {stage} training loss")
            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.step()
            total += float(loss.item())
            steps += 1
            if steps % args.log_interval == 0:
                print(json.dumps({"stage": stage, "epoch": epoch + 1, "step": steps,
                                  "loss": float(loss.item())}, sort_keys=True), flush=True)
        validation_set.set_epoch(epoch)
        validation_loss = mean_loss(model, validation_set)
        record = {"epoch": epoch + 1, "steps": steps, "mean_train_loss": total / max(steps, 1),
                  "validation_loss": validation_loss, "elapsed_seconds": time.time() - started}
        records.append(record)
        improved = validation_loss < best_loss - args.min_delta
        if improved or not best.is_file():
            best_loss = validation_loss
            best_epoch = epoch + 1
        metadata = {"stage": stage, "epoch": epoch + 1, "requested_epochs": epochs,
                    "best_validation_loss": best_loss, "best_epoch": best_epoch,
                    "config": config, "records": records}
        save_checkpoint(latest, model, metadata)
        if improved or not best.is_file():
            shutil.copy2(latest, best)
        write_json(state_path, {**metadata, "completed": epoch + 1 == epochs,
                                "latest_sha256": sha256_file(latest),
                                "best_sha256": sha256_file(best)})
        print(json.dumps(record, sort_keys=True), flush=True)
    return best, json.loads(state_path.read_text(encoding="utf-8"))


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-root", type=Path, required=True)
    parser.add_argument("--train-list", type=Path, required=True)
    parser.add_argument("--validation-list", type=Path, default=PACKAGE_ROOT / "configs/lists/rot_validate.txt")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stage-epochs", type=int, nargs=3, default=(3, 30, 40), metavar=("VM", "CVM", "SPCF"))
    parser.add_argument("--samples-per-epoch", type=int, default=10000)
    parser.add_argument("--validation-count", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--surface-points", type=int, default=32768)
    parser.add_argument("--vertex-samples", type=int, default=1024)
    parser.add_argument("--patch-size", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-delta", type=float, default=5e-7)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    jt.flags.use_cuda = 1 if args.device == "cuda" else 0
    for path in (args.mesh_root, args.train_list, args.validation_list):
        if not path.exists():
            raise PLRError(f"missing ROT input: {path}")
    if any(value <= 0 for value in (*args.stage_epochs, args.samples_per_epoch, args.batch_size,
                                    args.validation_batch_size, args.surface_points,
                                    args.patch_size)):
        raise PLRError("ROT counts and epoch values must be positive")
    if args.patch_size > args.surface_points or args.vertex_samples > args.surface_points:
        raise PLRError("invalid ROT patch/surface sampling sizes")
    if args.output_root.exists() and not args.resume:
        raise PLRError(f"ROT output root exists; pass --resume: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    train_keys = read_keys(args.train_list)
    validation_keys = read_keys(args.validation_list)
    if len(train_keys) != 15733:
        raise PLRError(f"expected 15733 ROT training keys, found {len(train_keys)}")
    previous = None
    stage_records = []
    for stage, epochs in zip(STAGES, args.stage_epochs):
        previous, state = train_stage(stage, epochs, previous, args, train_keys, validation_keys)
        stage_records.append({"stage": stage, "best_checkpoint": str(previous.resolve()),
                              "best_checkpoint_sha256": sha256_file(previous), "state": state})
    final_payload = jt.load(str(previous))
    final_state = final_payload.get("state_dict", final_payload)
    final_path = args.output_root / "spcf-final.pkl"
    temporary = args.output_root / "spcf-final.tmp.pkl"
    jt.save(final_state, str(temporary))
    os.replace(temporary, final_path)
    manifest = {
        "experiment": "NKAI-Jittor-StraightPCF-from-raw",
        "framework": "Jittor",
        "pytorch_runtime": False,
        "third_party_runtime": False,
        "train_list_sha256": sha256_file(args.train_list),
        "validation_list_sha256": sha256_file(args.validation_list),
        "config": {key: value for key, value in vars(args).items() if not isinstance(value, Path)},
        "stages": stage_records,
        "output_checkpoint": str(final_path.resolve()),
        "output_checkpoint_sha256": sha256_file(final_path),
    }
    write_json(args.output_root / "training_manifest.json", manifest)
    print(json.dumps({"output": str(final_path), "sha256": manifest["output_checkpoint_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PLRError as exc:
        print(f"ERROR: {exc}", flush=True)
        raise SystemExit(2)
