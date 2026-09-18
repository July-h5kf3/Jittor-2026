"""Isolated CPU sampler used to overlap mesh preparation with Jittor GPU work."""

from __future__ import annotations

import argparse
import pickle
import random
import sys

import numpy as np

from b_mesh_jittor import BMeshSampler
from combined_mesh_jittor import CombinedMeshSampler
from rank_mesh_jittor import RankMeshSampler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-format", choices=("bboard", "rank", "combined"), required=True)
    parser.add_argument("--mesh-root", required=True)
    parser.add_argument("--datalist", required=True)
    parser.add_argument("--patch-size", type=int, required=True)
    parser.add_argument("--patch-ratio", type=float, required=True)
    parser.add_argument("--patches-per-mesh", type=int, required=True)
    return parser.parse_args()


def merge(samples):
    return {
        "noisy": np.concatenate([sample["noisy"] for sample in samples], axis=0),
        "clean": np.concatenate([sample["clean"] for sample in samples], axis=0),
        "seeds": np.concatenate([sample["seeds"] for sample in samples], axis=0),
        "scale": np.concatenate([sample["scale"] for sample in samples], axis=0),
        "kind": sum((list(sample["kind"]) for sample in samples), []),
        "transform": np.concatenate([sample["transform"] for sample in samples], axis=0),
    }


def main():
    args = parse_args()
    sampler_class = {
        "rank": RankMeshSampler,
        "bboard": BMeshSampler,
        "combined": CombinedMeshSampler,
    }[args.dataset_format]
    sampler = sampler_class(
        args.mesh_root, args.datalist, patch_size=args.patch_size,
        patch_ratio=args.patch_ratio, patches_per_mesh=args.patches_per_mesh,
    )
    input_stream, output_stream = sys.stdin.buffer, sys.stdout.buffer
    while True:
        request = pickle.load(input_stream)
        if request is None:
            break
        mesh_indices, sample_seeds = request
        samples = []
        for mesh_index, seed in zip(mesh_indices, sample_seeds):
            random.seed(seed)
            np.random.seed(seed & 0xFFFFFFFF)
            samples.append(sampler.sample(mesh_index))
        pickle.dump(merge(samples), output_stream, protocol=pickle.HIGHEST_PROTOCOL)
        output_stream.flush()


if __name__ == "__main__":
    main()
