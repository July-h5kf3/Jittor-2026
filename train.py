#!/usr/bin/env python3
"""Jittor-native 3DMambaIPF base training and expert/DCD continuation."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import jittor as jt
import numpy as np
from jittor import optim

from plr3d.backend import add_device_argument, configure_device
from plr3d.data import PLRPatchDataset, read_keys
from plr3d.io import (
    PLRError,
    load_model_checkpoint,
    save_jittor_checkpoint_new,
    sha256_file,
    write_json_new,
)
from plr3d.losses import plr_training_loss
from plr3d.models import DenoiseNet


def load_sigma_map(path: Path) -> Dict[str, Tuple[float, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("by_category", payload)
    if not isinstance(records, dict) or not records:
        raise PLRError("sigma map must be a non-empty mapping")
    result = {}
    for category, record in records.items():
        if isinstance(record, dict):
            lower = float(record["min"])
            upper = float(record["max"])
        else:
            lower, upper = map(float, record)
        if not 0.0 < lower <= upper < 0.1:
            raise PLRError(f"invalid sigma range for {category}: {lower}, {upper}")
        result[str(category)] = (lower, upper)
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    initialization = parser.add_mutually_exclusive_group(required=True)
    initialization.add_argument("--checkpoint", type=Path)
    initialization.add_argument(
        "--random-init", action="store_true",
        help="train a base model from deterministic Jittor initialization",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-list", type=Path, required=True)
    parser.add_argument("--sigma-map-json", type=Path)
    parser.add_argument("--sigma-min", type=float)
    parser.add_argument("--sigma-max", type=float)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint-prefix", default="plr")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--samples-per-shape", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--grad-accum-steps", type=int, default=1,
        help="micro-batches per optimizer update; use 4 to reproduce four-rank global batches on one GPU",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=1000)
    parser.add_argument("--patch-ratio", type=float, default=1.2)
    parser.add_argument("--gaussian-probability", type=float, default=0.0)
    parser.add_argument("--outlier-fraction", type=float, default=0.0)
    parser.add_argument("--outlier-scale", type=float, default=3.0)
    parser.add_argument("--scale-min", type=float, default=1.0)
    parser.add_argument("--scale-max", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--dcd-alpha", type=float, default=100.0)
    parser.add_argument("--dcd-n-lambda", type=float, default=0.5)
    parser.add_argument("--dcd-weight", type=float, default=0.5)
    parser.add_argument("--frame-knn", type=int, default=32)
    parser.add_argument("--num-modules", type=int, default=4)
    parser.add_argument("--noise-decay", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=20262731)
    parser.add_argument("--seed-namespace", default="plr003-tail-item")
    parser.add_argument("--expected-train-count", type=int)
    parser.add_argument("--expected-steps-per-epoch", type=int)
    parser.add_argument("--log-interval", type=int, default=50)
    add_device_argument(parser)
    return parser.parse_args(argv)


def _as_jittor_batch(batch):
    result = {}
    for key, value in batch.items():
        result[key] = value if isinstance(value, jt.Var) else jt.array(value)
    return result


def _barrier() -> None:
    if bool(getattr(jt, "in_mpi", False)):
        value = jt.array([1.0]).mpi_all_reduce("sum")
        value.sync()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_device(args.device, jt)
    rank = int(getattr(jt, "rank", 0) or 0)
    world_size = int(getattr(jt, "world_size", 1) or 1)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    jt.set_global_seed(args.seed + rank)

    required_paths = [args.data_root, args.train_list]
    if args.checkpoint is not None:
        required_paths.append(args.checkpoint)
    for path in required_paths:
        if not path.exists():
            raise PLRError(f"missing training input: {path}")
    if args.output_root.exists():
        raise PLRError(f"output root already exists: {args.output_root}")
    if (
        args.epochs <= 0
        or args.batch_size <= 0
        or args.grad_accum_steps <= 0
        or args.dcd_weight < 0.0
    ):
        raise PLRError(
            "epochs, batch size and accumulation must be positive; DCD weight must be non-negative"
        )

    keys = read_keys(args.train_list)
    if args.expected_train_count is not None and len(keys) != args.expected_train_count:
        raise PLRError(
            f"expected {args.expected_train_count} keys, found {len(keys)}"
        )
    categories = {key.split("/")[1] for key in keys}
    if args.sigma_map_json is not None:
        sigma_map = load_sigma_map(args.sigma_map_json)
        sigma_map_sha = sha256_file(args.sigma_map_json)
    else:
        if args.sigma_min is None or args.sigma_max is None:
            raise PLRError("provide --sigma-map-json or both --sigma-min/--sigma-max")
        sigma_map = {
            category: (float(args.sigma_min), float(args.sigma_max))
            for category in categories
        }
        sigma_map_sha = None
    if set(sigma_map) != categories:
        raise PLRError("sigma-map categories do not match the training list")

    if rank == 0:
        args.output_root.mkdir(parents=True)
    _barrier()

    model = DenoiseNet(
        frame_knn=args.frame_knn,
        num_modules=args.num_modules,
        noise_decay=args.noise_decay,
    )
    if args.checkpoint is not None:
        base_metadata = load_model_checkpoint(model, args.checkpoint)
        base_checkpoint_sha256 = sha256_file(args.checkpoint)
        initialization_mode = "checkpoint"
    else:
        base_metadata = {"initialization": "deterministic Jittor random initialization"}
        base_checkpoint_sha256 = None
        initialization_mode = "random"
    if world_size > 1:
        model.mpi_param_broadcast(root=0)
    model.train()

    dataset = PLRPatchDataset(
        data_root=args.data_root,
        keys=keys,
        samples_per_shape=args.samples_per_shape,
        patch_size=args.patch_size,
        patch_ratio=args.patch_ratio,
        sigma_by_category=sigma_map,
        gaussian_probability=args.gaussian_probability,
        outlier_fraction=args.outlier_fraction,
        outlier_scale=args.outlier_scale,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        seed=args.seed,
        seed_namespace=args.seed_namespace,
    )
    dataset.set_attrs(
        batch_size=args.batch_size,
        total_len=len(dataset),
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    config = {
        "framework": "Jittor",
        "pytorch_runtime": False,
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "initialization_mode": initialization_mode,
        "train_list_sha256": sha256_file(args.train_list),
        "sigma_map_sha256": sigma_map_sha,
        "sigma_by_category": {
            key: {"min": value[0], "max": value[1]}
            for key, value in sorted(sigma_map.items())
        },
        "train_count": len(keys),
        "samples_per_shape": args.samples_per_shape,
        "batch_size_per_process": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "effective_batch_size": args.batch_size * args.grad_accum_steps * world_size,
        "world_size": world_size,
        "epochs": args.epochs,
        "patch_size": args.patch_size,
        "patch_ratio": args.patch_ratio,
        "gaussian_probability": args.gaussian_probability,
        "outlier_fraction": args.outlier_fraction,
        "outlier_scale": args.outlier_scale,
        "scale_min": args.scale_min,
        "scale_max": args.scale_max,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "dcd_alpha": args.dcd_alpha,
        "dcd_n_lambda": args.dcd_n_lambda,
        "dcd_weight": args.dcd_weight,
        "seed": args.seed,
        "seed_namespace": args.seed_namespace,
        "architecture": model.architecture_metadata(),
        "base_metadata": base_metadata,
    }

    epoch_records = []
    global_step = 0
    training_started = time.time()
    for epoch_index in range(args.epochs):
        dataset.set_epoch(epoch_index)
        epoch_started = time.time()
        loss_sum = 0.0
        dcd_sum = 0.0
        stage_sum = np.zeros(args.num_modules, dtype=np.float64)
        micro_batch_count = 0
        optimizer_steps_epoch = 0
        total_samples_per_rank = int(np.ceil(len(keys) * args.samples_per_shape / world_size))
        expected_micro_batches = int(np.ceil(total_samples_per_rank / args.batch_size))
        optimizer.zero_grad()
        for batch in dataset:
            micro_batch_count += 1
            batch = _as_jittor_batch(batch)
            loss, stage_losses, dcd = plr_training_loss(
                model,
                batch,
                dcd_alpha=args.dcd_alpha,
                dcd_n_lambda=args.dcd_n_lambda,
                dcd_weight=args.dcd_weight,
                outlier_fraction=args.outlier_fraction,
                outlier_scale=args.outlier_scale,
            )
            if not bool(jt.isfinite(loss).all().item()):
                raise PLRError("non-finite PLR training loss")
            group_start = ((micro_batch_count - 1) // args.grad_accum_steps) * args.grad_accum_steps
            group_size = min(
                args.grad_accum_steps, expected_micro_batches - group_start
            )
            optimizer.backward(loss / max(group_size, 1))
            should_step = (
                micro_batch_count % args.grad_accum_steps == 0
                or micro_batch_count == expected_micro_batches
            )
            if should_step:
                optimizer.clip_grad_norm(args.grad_clip, 2)
                optimizer.step()
                optimizer.zero_grad()
                optimizer_steps_epoch += 1
                global_step += 1

            loss_value = float(loss.item())
            dcd_value = float(dcd.item())
            stage_value = np.asarray(stage_losses.numpy(), dtype=np.float64)
            loss_sum += loss_value
            dcd_sum += dcd_value
            stage_sum += stage_value
            if rank == 0 and (
                micro_batch_count % args.log_interval == 0
                or micro_batch_count == expected_micro_batches
            ):
                print(
                    json.dumps(
                        {
                            "epoch": epoch_index + 1,
                            "micro_batch": micro_batch_count,
                            "optimizer_step": optimizer_steps_epoch,
                            "global_step": global_step,
                            "loss": loss_value,
                            "dcd": dcd_value,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        if micro_batch_count != expected_micro_batches:
            raise PLRError(
                f"expected {expected_micro_batches} micro-batches, found {micro_batch_count}"
            )
        if (
            args.expected_steps_per_epoch is not None
            and optimizer_steps_epoch != args.expected_steps_per_epoch
        ):
            raise PLRError(
                f"expected {args.expected_steps_per_epoch} optimizer steps, "
                f"found {optimizer_steps_epoch}"
            )
        record = {
            "epoch": epoch_index + 1,
            "steps": optimizer_steps_epoch,
            "micro_batches": micro_batch_count,
            "mean_loss": loss_sum / max(micro_batch_count, 1),
            "mean_dcd": dcd_sum / max(micro_batch_count, 1),
            "mean_stage_losses": (stage_sum / max(micro_batch_count, 1)).tolist(),
            "elapsed_seconds": time.time() - epoch_started,
        }
        epoch_records.append(record)
        if rank == 0:
            checkpoint_path = args.output_root / (
                f"{args.checkpoint_prefix}-epoch{epoch_index + 1:02d}.pkl"
            )
            save_jittor_checkpoint_new(
                model,
                checkpoint_path,
                {
                    "experiment": args.checkpoint_prefix,
                    "epoch": epoch_index + 1,
                    "global_step": global_step,
                    "config": config,
                    "epoch_record": record,
                },
            )
            print(json.dumps(record, sort_keys=True), flush=True)
        _barrier()

    if rank == 0:
        summary = {
            "experiment": args.checkpoint_prefix,
            "framework": "Jittor",
            "pytorch_runtime": False,
            "config": config,
            "global_step": global_step,
            "elapsed_seconds": time.time() - training_started,
            "epochs": epoch_records,
        }
        write_json_new(args.output_root / "training_summary.json", summary)
    _barrier()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PLRError as exc:
        print(f"ERROR: {exc}", flush=True)
        raise SystemExit(2)
