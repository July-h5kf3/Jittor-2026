"""Six-GPU native Jittor fine-tuning for B-board IterativePFN."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
import pickle
import random
import subprocess
import sys
import time
from pathlib import Path

import jittor as jt
import numpy as np

from b_mesh_jittor import BMeshSampler
from combined_mesh_jittor import CombinedMeshSampler
from rank_mesh_jittor import RankMeshSampler
from model import IterativePFN, export_converted_weights, load_converted_weights
from train_loss import joint_metric_loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-root", required=True)
    parser.add_argument("--datalist", required=True)
    parser.add_argument("--dataset-format", choices=("bboard", "rank", "combined"), default="bboard")
    parser.add_argument("--init-weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--patch-size", type=int, default=1000)
    parser.add_argument("--patch-ratio", type=float, default=1.2)
    parser.add_argument("--patches-per-mesh", type=int, default=4)
    parser.add_argument(
        "--meshes-per-step", type=int, default=1,
        help="Concatenate this many complete mesh samples per rank/optimizer step",
    )
    parser.add_argument("--metric-weight", type=float, default=0.2)
    parser.add_argument("--metric-warmup-epochs", type=float, default=2.0)
    parser.add_argument("--noise-decay", type=float, default=4.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument("--resume-weights", default="", help="Resume model weights from an exported epoch NPZ")
    parser.add_argument("--start-epoch", type=int, default=0, help="Completed epochs before this launch")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--resume-state", default="", help="Resume native Jittor model+Adam state")
    parser.add_argument("--use-tensorcore", action="store_true", help="Enable TF32/tensor-core kernels")
    parser.add_argument(
        "--sample-workers", type=int, default=1,
        help="CPU threads used to prepare meshes within one optimizer step",
    )
    parser.add_argument(
        "--prefetch-batches", type=int, default=0, choices=(0, 1, 2),
        help="0=off, 1=thread, 2=isolated CPU process for next-batch preparation",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    jt.set_global_seed(seed)


def shard_order(length, epoch, seed, rank, world_size):
    rng = np.random.RandomState(seed + epoch)
    order = rng.permutation(length).tolist()
    padded = int(math.ceil(length / world_size) * world_size)
    order.extend(order[: padded - length])
    return order[rank:padded:world_size]


def merge_mesh_samples(samples):
    """Concatenate complete per-mesh patch batches without dropping coverage."""
    return {
        "noisy": np.concatenate([sample["noisy"] for sample in samples], axis=0),
        "clean": np.concatenate([sample["clean"] for sample in samples], axis=0),
        "seeds": np.concatenate([sample["seeds"] for sample in samples], axis=0),
        "scale": np.concatenate([sample["scale"] for sample in samples], axis=0),
        "kind": sum((list(sample["kind"]) for sample in samples), []),
        "transform": np.concatenate([sample["transform"] for sample in samples], axis=0),
    }


def sample_one(sampler, mesh_index, seed):
    # Only the dedicated sampling path touches NumPy/Python RNG state. Jittor's
    # RNG remains owned by the main training thread while GPU work runs.
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    return sampler.sample(mesh_index)


def prepare_mesh_batch(sampler, mesh_indices, seeds_for_meshes):
    samples = [
        sample_one(sampler, mesh_index, sample_seed)
        for mesh_index, sample_seed in zip(mesh_indices, seeds_for_meshes)
    ]
    return merge_mesh_samples(samples)


class ProcessBatchPrefetcher:
    """Persistent sampler process; avoids the GIL and never touches CUDA."""

    def __init__(self, args):
        command = [
            sys.executable, str(Path(__file__).with_name("sampling_worker.py")),
            "--dataset-format", args.dataset_format, "--mesh-root", args.mesh_root,
            "--datalist", args.datalist, "--patch-size", str(args.patch_size),
            "--patch-ratio", str(args.patch_ratio),
            "--patches-per-mesh", str(args.patches_per_mesh),
        ]
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=sys.stderr, bufsize=0,
        )

    def submit(self, mesh_indices, sample_seeds):
        pickle.dump((mesh_indices, sample_seeds), self.process.stdin, protocol=pickle.HIGHEST_PROTOCOL)
        self.process.stdin.flush()

    def result(self):
        try:
            return pickle.load(self.process.stdout)
        except EOFError as error:
            raise RuntimeError("sampling subprocess exited with status %s" % self.process.poll()) from error

    def close(self):
        if self.process.poll() is None:
            pickle.dump(None, self.process.stdin, protocol=pickle.HIGHEST_PROTOCOL)
            self.process.stdin.flush()
            self.process.wait(timeout=30)


def optimizer_state_numpy(optimizer):
    groups = []
    for group in optimizer.param_groups:
        groups.append({
            "m": [value.numpy() for value in group["m"]],
            "values": [value.numpy() for value in group["values"]],
            "lr": group.get("lr", optimizer.lr),
        })
    return {"n_step": optimizer.n_step, "lr": optimizer.lr, "groups": groups}


def restore_optimizer_state(optimizer, state):
    optimizer.n_step = int(state["n_step"])
    optimizer.lr = float(state["lr"])
    if len(state["groups"]) != len(optimizer.param_groups):
        raise RuntimeError("optimizer group count mismatch")
    for group, saved in zip(optimizer.param_groups, state["groups"]):
        if len(saved["m"]) != len(group["m"]):
            raise RuntimeError("optimizer parameter count mismatch")
        for target, value in zip(group["m"], saved["m"]):
            target.assign(value)
        for target, value in zip(group["values"], saved["values"]):
            target.assign(value)
        group["lr"] = float(saved["lr"])


def save_epoch(model, optimizer, output, epoch, global_step, args, elapsed):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / ("denoisenet-jittor-b-epoch%02d.npz" % epoch)
    # Calling numpy() on all ranks keeps MPI execution symmetric. Only rank 0
    # writes the common file.
    export_converted_weights(model, str(checkpoint) if jt.rank == 0 else None)
    optimizer_state = optimizer_state_numpy(optimizer)
    if jt.rank == 0:
        with (output / ("training-state-epoch%02d.pkl" % epoch)).open("wb") as handle:
            pickle.dump({
                "weights": checkpoint.name, "optimizer": optimizer_state,
                "epoch": epoch, "global_step": global_step,
            }, handle, protocol=pickle.HIGHEST_PROTOCOL)
        metadata = {
            "framework": "jittor",
            "epoch": epoch,
            "global_step": global_step,
            "elapsed_seconds": elapsed,
            "world_size": jt.world_size,
            "checkpoint": checkpoint.name,
            "args": vars(args),
        }
        (output / ("epoch%02d.json" % epoch)).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    jt.sync_all(True)


def main():
    args = parse_args()
    # Multiple MPI ranks compiling a new fused shape with Jittor's internal
    # parallel op compiler can race in UCX/relay management. Keep GPU/MPI
    # execution parallel but serialize per-rank JIT compilation; cached kernels
    # make this a one-time startup cost.
    jt.flags.use_parallel_op_compiler = 0
    jt.flags.use_cuda = 1
    # Optional on Ampere: benchmark before enabling for a formal run.
    jt.flags.use_tensorcore = int(args.use_tensorcore)
    rank, world_size = jt.rank, jt.world_size
    set_seed(args.seed + rank * 100003)

    sampler_class = {
        "rank": RankMeshSampler,
        "bboard": BMeshSampler,
        "combined": CombinedMeshSampler,
    }[args.dataset_format]
    sampler = sampler_class(
        args.mesh_root,
        args.datalist,
        patch_size=args.patch_size,
        patch_ratio=args.patch_ratio,
        patches_per_mesh=args.patches_per_mesh,
    )
    model = IterativePFN()
    resume_state = None
    if args.resume_state:
        with open(args.resume_state, "rb") as handle:
            resume_state = pickle.load(handle)
        load_converted_weights(model, str(Path(args.resume_state).parent / resume_state["weights"]))
    else:
        load_path = args.resume_weights or args.init_weights
        load_converted_weights(model, load_path)
    model.train()
    optimizer = jt.optim.Adam(model.parameters(), lr=args.lr)
    if resume_state is not None:
        restore_optimizer_state(optimizer, resume_state["optimizer"])
        if args.start_epoch != int(resume_state["epoch"]):
            raise RuntimeError("start-epoch does not match resume-state")
        # Preserve Adam moments and step exactly, while allowing the explicit
        # stage schedule to lower the learning rate at epochs 10 and 20.
        optimizer.lr = args.lr
        for group in optimizer.param_groups:
            group["lr"] = args.lr
    probe_parameter = model.feature_nets[0].linear3.weight
    probe_initial = probe_parameter.numpy().copy()

    if args.meshes_per_step < 1:
        raise ValueError("meshes-per-step must be positive")
    samples_per_global_step = world_size * args.meshes_per_step
    steps_full = int(math.ceil(len(sampler) / samples_per_global_step))
    steps_epoch = min(steps_full, args.max_steps_per_epoch) if args.max_steps_per_epoch else steps_full
    if rank == 0:
        print(json.dumps({
            "event": "start", "framework": "jittor", "meshes": len(sampler),
            "world_size": world_size, "steps_per_epoch": steps_epoch,
            "meshes_per_step_per_rank": args.meshes_per_step,
            "original_occurrences_per_epoch": len(sampler),
            "padded_occurrences_per_epoch": steps_full * samples_per_global_step,
            "patch_batch_per_rank": args.patches_per_mesh * args.meshes_per_step,
            "effective_patch_batch": args.patches_per_mesh * samples_per_global_step,
        }), flush=True)

    global_step = int(resume_state["global_step"]) if resume_state is not None else 0
    started = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # Pad to complete global optimizer steps, then give each rank consecutive
        # groups of meshes_per_step. Every original occurrence is covered once;
        # only the unavoidable tail padding can repeat samples.
        rng = np.random.RandomState(args.seed + epoch)
        global_order = rng.permutation(len(sampler)).tolist()
        padded = steps_full * samples_per_global_step
        global_order.extend(global_order[: padded - len(global_order)])
        order = []
        for step in range(steps_epoch):
            base = step * samples_per_global_step + rank * args.meshes_per_step
            order.append(global_order[base : base + args.meshes_per_step])
        epoch_total = epoch_legacy = epoch_metric = 0.0
        epoch_data_seconds = epoch_compute_seconds = 0.0
        epoch_start = time.time()
        def prepare_step(local_step):
            mesh_indices = order[local_step]
            seeds_for_meshes = [
                args.seed + epoch * 1000003 + int(mesh_index) * 17 + rank
                for mesh_index in mesh_indices
            ]
            return prepare_mesh_batch(sampler, mesh_indices, seeds_for_meshes)

        prefetch_executor = None
        process_prefetcher = None
        prefetched = None
        if args.prefetch_batches == 1:
            # One worker preserves the exact sequential RNG recipe while
            # overlapping CPU preparation with GPU computation.
            prefetch_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mesh-prefetch")
            prefetched = prefetch_executor.submit(prepare_step, 0)
        elif args.prefetch_batches == 2:
            process_prefetcher = ProcessBatchPrefetcher(args)
            first_indices = order[0]
            first_seeds = [
                args.seed + epoch * 1000003 + int(mesh_index) * 17 + rank
                for mesh_index in first_indices
            ]
            process_prefetcher.submit(first_indices, first_seeds)

        for local_step, mesh_indices in enumerate(order):
            # Make each (epoch, mesh, rank) sample deterministic and independent
            # of timing or process scheduling.
            data_started = time.time()
            if args.sample_workers != 1:
                raise ValueError("sample-workers > 1 requires per-sample RNGs for reproducibility")
            if process_prefetcher is not None:
                batch = process_prefetcher.result()
                if local_step + 1 < len(order):
                    next_indices = order[local_step + 1]
                    next_seeds = [
                        args.seed + epoch * 1000003 + int(mesh_index) * 17 + rank
                        for mesh_index in next_indices
                    ]
                    process_prefetcher.submit(next_indices, next_seeds)
            elif prefetched is not None:
                batch = prefetched.result()
                if local_step + 1 < len(order):
                    prefetched = prefetch_executor.submit(prepare_step, local_step + 1)
            else:
                batch = prepare_step(local_step)
            # Match the former sequential sample_one(set_seed(...)) behavior:
            # the adaptive Jittor target RNG starts from the final mesh seed.
            final_mesh_seed = args.seed + epoch * 1000003 + int(mesh_indices[-1]) * 17 + rank
            jt.set_global_seed(final_mesh_seed)
            epoch_data_seconds += time.time() - data_started
            compute_started = time.time()
            noisy = jt.array(batch["noisy"])
            clean = jt.array(batch["clean"])
            seeds = jt.array(batch["seeds"])
            scale = jt.array(batch["scale"])
            transform = jt.array(batch["transform"])
            progress_epochs = epoch + float(local_step) / max(steps_epoch, 1)
            warm = max(args.metric_warmup_epochs, 1e-12)
            metric_weight = args.metric_weight * min(max(progress_epochs / warm, 0.0), 1.0)
            total, legacy, metric, _ = joint_metric_loss(
                model, noisy, clean, seeds, scale, batch["kind"], transform,
                noise_decay=args.noise_decay, metric_weight=metric_weight,
            )
            optimizer.zero_grad()
            optimizer.backward(total)
            if args.grad_clip > 0:
                optimizer.clip_grad_norm(args.grad_clip)
            optimizer.step()

            # All ranks evaluate these scalars to keep execution symmetric.
            total_value = float(total.item())
            legacy_value = float(legacy.item())
            metric_value = float(metric.item())
            epoch_compute_seconds += time.time() - compute_started
            epoch_total += total_value
            epoch_legacy += legacy_value
            epoch_metric += metric_value
            global_step += 1
            if rank == 0 and (local_step == 0 or (local_step + 1) % args.log_every == 0):
                elapsed = max(time.time() - epoch_start, 1e-9)
                print(json.dumps({
                    "event": "step", "epoch": epoch + 1, "step": local_step + 1,
                    "steps": steps_epoch, "global_step": global_step,
                    "loss": total_value, "legacy": legacy_value, "metric": metric_value,
                    "metric_weight": metric_weight, "steps_per_second": (local_step + 1) / elapsed,
                    "meshes_per_second": (local_step + 1) * args.meshes_per_step / elapsed,
                    "data_seconds_per_step": epoch_data_seconds / (local_step + 1),
                    "compute_seconds_per_step": epoch_compute_seconds / (local_step + 1),
                }), flush=True)

        if prefetch_executor is not None:
            prefetch_executor.shutdown(wait=True)
        if process_prefetcher is not None:
            process_prefetcher.close()

        denom = max(len(order), 1)
        if rank == 0:
            print(json.dumps({
                "event": "epoch", "epoch": epoch + 1,
                "loss_mean_rank0": epoch_total / denom,
                "legacy_mean_rank0": epoch_legacy / denom,
                "metric_mean_rank0": epoch_metric / denom,
                "data_seconds_rank0": epoch_data_seconds,
                "compute_seconds_rank0": epoch_compute_seconds,
                "seconds": time.time() - epoch_start,
            }), flush=True)
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            save_epoch(model, optimizer, args.output, epoch + 1, global_step, args, time.time() - started)

    probe_final = probe_parameter.numpy()
    probe_delta = float(np.max(np.abs(probe_final - probe_initial)))
    if rank == 0:
        print(json.dumps({"event": "gradient_check", "max_parameter_delta": probe_delta}), flush=True)
    if not np.isfinite(probe_delta) or probe_delta <= 0:
        raise RuntimeError("training completed without a finite parameter update")


if __name__ == "__main__":
    main()
