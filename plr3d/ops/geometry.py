"""Geometry operators used by the Jittor-native 3DMambaIPF port."""

from __future__ import annotations

from typing import Optional, Tuple

import jittor as jt


def gather_points(points: jt.Var, indices: jt.Var) -> jt.Var:
    """Gather batched points/features.

    Args:
        points: ``(B, N, C)``.
        indices: ``(B, ..., K)`` integer indices into axis 1.
    Returns:
        ``(B, ..., K, C)``.
    """
    if points.ndim != 3 or indices.ndim < 2:
        raise ValueError("gather_points expects points=(B,N,C) and batched indices")
    batch, count, channels = points.shape
    if indices.shape[0] != batch:
        raise ValueError("batch dimension mismatch in gather_points")
    tail_shape = tuple(indices.shape[1:])
    # ACL's advanced Index operator does not support the flattened indexing
    # form used by the CUDA path.  Jittor ACL provides Gather instead.  Move
    # the point axis to dim 2 so batch and channel dimensions can be preserved
    # while the arbitrary query/K tail is flattened for Gather.
    flat_count = 1
    for size in tail_shape:
        flat_count *= size
    source = points.permute(0, 2, 1)
    gather_index = indices.int32().reshape(batch, 1, flat_count).broadcast(
        (batch, channels, flat_count)
    )
    gathered = jt.gather(source, 2, gather_index)
    return gathered.reshape((batch, channels) + tail_shape).permute(
        0, *range(2, 2 + len(tail_shape)), 1
    )


def knn_indices(query: jt.Var, reference: jt.Var, k: int) -> Tuple[jt.Var, jt.Var]:
    """Return squared distances and indices for batched KNN.

    Jittor's CUDA KNN follows the same ``query, reference`` convention used here.
    Both tensors must be float32 coordinate/feature arrays of shape ``(B,N,C)``.
    """
    if query.ndim != 3 or reference.ndim != 3:
        raise ValueError("knn_indices expects batched rank-3 tensors")
    if query.shape[0] != reference.shape[0] or query.shape[2] != reference.shape[2]:
        raise ValueError("KNN batch/feature dimensions must match")
    if k <= 0 or k > reference.shape[1]:
        raise ValueError(f"invalid k={k} for reference count {reference.shape[1]}")
    if query.dtype != "float32":
        query = query.float32()
    if reference.dtype != "float32":
        reference = reference.float32()
    # ``jt.misc.knn`` is a CUDA custom operator.  The pinned ACL backend has
    # no ACL KNN implementation, while the regular pairwise distance/topk
    # composition is supported and stays on the selected ACL device.
    if query.shape[2] == 3 and not getattr(jt.flags, "use_acl", 0):
        distances, indices = jt.misc.knn(query, reference, k)
    else:
        pairwise = ((query.unsqueeze(2) - reference.unsqueeze(1)) ** 2).sum(dim=-1)
        distances, indices = jt.topk(pairwise, k=k, dim=-1, largest=False)
    return distances, indices.int32()


def knn_points(
    query: jt.Var,
    reference: jt.Var,
    k: int,
    return_nn: bool = True,
) -> Tuple[jt.Var, jt.Var, Optional[jt.Var]]:
    _, indices = knn_indices(query, reference, k)
    gathered = gather_points(reference, indices)
    distances = ((query.unsqueeze(2) - gathered) ** 2).sum(dim=-1)
    neighbours = gathered if return_nn else None
    return distances, indices, neighbours


def farthest_point_sampling(points: jt.Var, count: int) -> Tuple[jt.Var, jt.Var]:
    """Deterministic fixed-start farthest-point sampling.

    This intentionally starts from row 0, matching the frozen PLR inference path.
    ``points`` has shape ``(B,N,3)``. The returned indices are ``int32``.
    """
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("farthest_point_sampling expects (B,N,3)")
    batch, point_count, _ = points.shape
    if count <= 0 or count > point_count:
        raise ValueError(f"sample count must be in [1,{point_count}], got {count}")

    sampled_batches = []
    index_batches = []
    for batch_index in range(batch):
        cloud = points[batch_index]
        minimum_distance = jt.ones((point_count,), dtype=points.dtype) * float("inf")
        selected = []
        current = 0
        for _ in range(count):
            selected.append(current)
            centroid = cloud[current]
            squared_distance = ((cloud - centroid) ** 2).sum(dim=-1)
            minimum_distance = jt.minimum(minimum_distance, squared_distance)
            current = int(jt.argmax(minimum_distance, dim=0)[0].item())
        indices = jt.array(selected).int32()
        sampled_batches.append(cloud[indices].unsqueeze(0))
        index_batches.append(indices.unsqueeze(0))
    return jt.concat(sampled_batches, dim=0), jt.concat(index_batches, dim=0)
