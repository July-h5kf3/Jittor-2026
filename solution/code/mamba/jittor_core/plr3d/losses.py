"""Jittor-native PLR nearest-neighbour and density-aware Chamfer losses."""

from __future__ import annotations

from typing import Dict, Tuple

import jittor as jt

from .ops.geometry import knn_points


def progressive_noise(
    target: jt.Var,
    std: jt.Var,
    domain: jt.Var,
    outlier_fraction: float,
    outlier_scale: float,
) -> jt.Var:
    scale = std.reshape(std.shape[0], 1, 1)
    gaussian = jt.randn(target.shape) * scale
    uniform = jt.clamp(jt.rand(target.shape), 1e-7, 1.0 - 1e-7) - 0.5
    uniform_sign = jt.where(uniform >= 0.0, jt.ones_like(uniform), -jt.ones_like(uniform))
    laplace = -scale * uniform_sign * jt.log(1.0 - 2.0 * jt.abs(uniform))
    if outlier_fraction > 0.0:
        outlier_mask = jt.rand((target.shape[0], target.shape[1], 1)) < outlier_fraction
        laplace = jt.where(outlier_mask, laplace * outlier_scale, laplace)
    domain_mask = domain.reshape(domain.shape[0], 1, 1).bool()
    return jt.where(domain_mask, laplace, gaussian).float32()


def adaptive_nn_components(
    model,
    batch: Dict[str, jt.Var],
    outlier_fraction: float,
    outlier_scale: float,
) -> Tuple[jt.Var, jt.Var, jt.Var]:
    noisy = batch["pcl_noisy"].float32()
    clean = batch["pcl_clean"].float32()
    seeds = batch["pcl_seeds"].float32()
    std = batch["pcl_std"].float32()
    domain = batch["domain"]

    batch_size, noisy_count, _ = noisy.shape
    clean_count = clean.shape[1]
    seed_noisy = seeds.unsqueeze(1).broadcast((batch_size, noisy_count, 3))
    seed_clean = seeds.unsqueeze(1).broadcast((batch_size, clean_count, 3))

    seed_distance_sq = ((noisy - seed_noisy) ** 2).sum(dim=-1)
    max_seed_distance_sq = seed_distance_sq.max(dim=1, keepdims=True)
    max_seed_distance_sq = jt.maximum(max_seed_distance_sq, 1e-38)
    seed_weights = jt.exp(-seed_distance_sq / (max_seed_distance_sq / 9.0))
    seed_weights = seed_weights / jt.maximum(
        seed_weights.sum(dim=1, keepdims=True), 1e-38
    )

    current = noisy - seed_noisy
    clean_centered = clean - seed_clean
    current_std = std
    projection = None
    stage_losses = []
    predicted_displacement = None
    for stage, feature_net in enumerate(model.feature_nets):
        if stage > 0:
            current = current + predicted_displacement
        predicted_displacement, projection = feature_net(current, projection)
        if stage < len(model.feature_nets) - 1 and model.noise_decay != 1.0:
            current_std = current_std / model.noise_decay
            target = clean_centered + progressive_noise(
                clean_centered,
                current_std,
                domain,
                outlier_fraction,
                outlier_scale,
            )
        else:
            target = clean_centered
        _, _, nearest = knn_points(current, target, 1, return_nn=True)
        if nearest is None:
            raise RuntimeError("nearest-neighbour lookup returned no points")
        target_displacement = nearest.reshape(batch_size, noisy_count, 3) - current
        squared_error = ((predicted_displacement - target_displacement) ** 2).sum(
            dim=-1
        )
        stage_losses.append((seed_weights * squared_error).sum(dim=1).mean())

    if predicted_displacement is None:
        raise RuntimeError("model has no feature stages")
    endpoint = current + predicted_displacement
    return jt.concat([value.reshape(1) for value in stage_losses], dim=0), endpoint, clean_centered


def _gather_counts(counts: jt.Var, indices: jt.Var) -> jt.Var:
    batch, count = counts.shape
    base = (jt.arange(batch).int32() * count).reshape(batch, 1)
    return counts.reshape(-1)[(indices.int32() + base).reshape(-1)].reshape(indices.shape)


def density_aware_chamfer(
    prediction: jt.Var,
    target: jt.Var,
    alpha: float = 100.0,
    n_lambda: float = 0.5,
) -> jt.Var:
    if prediction.ndim != 3 or target.ndim != 3:
        raise ValueError("DCD inputs must be batched point sets")
    if prediction.shape[0] != target.shape[0] or prediction.shape[2] != target.shape[2]:
        raise ValueError("DCD batch/feature dimensions must match")
    if alpha <= 0.0 or not 0.0 <= n_lambda <= 1.0:
        raise ValueError("invalid DCD hyperparameters")

    dist_t2p, idx_t2p, _ = knn_points(target, prediction, 1, return_nn=False)
    dist_p2t, idx_p2t, _ = knn_points(prediction, target, 1, return_nn=False)
    dist_t2p = dist_t2p[:, :, 0]
    dist_p2t = dist_p2t[:, :, 0]
    idx_t2p = idx_t2p[:, :, 0].int32()
    idx_p2t = idx_p2t[:, :, 0].int32()

    batch, prediction_count, _ = prediction.shape
    target_count = target.shape[1]
    count_prediction = jt.zeros((batch, prediction_count), dtype="float32")
    count_prediction = count_prediction.scatter_(
        1,
        idx_t2p,
        jt.ones(idx_t2p.shape, dtype="float32"),
        reduce="add",
    )
    weight_t2p = _gather_counts(count_prediction, idx_t2p).stop_grad() ** n_lambda
    weight_t2p = (1.0 / (weight_t2p + 1e-6)) * (
        float(target_count) / float(prediction_count)
    )

    count_target = jt.zeros((batch, target_count), dtype="float32")
    count_target = count_target.scatter_(
        1,
        idx_p2t,
        jt.ones(idx_p2t.shape, dtype="float32"),
        reduce="add",
    )
    weight_p2t = _gather_counts(count_target, idx_p2t).stop_grad() ** n_lambda
    weight_p2t = (1.0 / (weight_p2t + 1e-6)) * (
        float(prediction_count) / float(target_count)
    )

    loss_t2p = (1.0 - jt.exp(-alpha * dist_t2p) * weight_t2p).mean(dim=1)
    loss_p2t = (1.0 - jt.exp(-alpha * dist_p2t) * weight_p2t).mean(dim=1)
    return ((loss_t2p + loss_p2t) * 0.5).mean()


def plr_training_loss(
    model,
    batch: Dict[str, jt.Var],
    dcd_alpha: float,
    dcd_n_lambda: float,
    dcd_weight: float,
    outlier_fraction: float = 0.0,
    outlier_scale: float = 3.0,
) -> Tuple[jt.Var, jt.Var, jt.Var]:
    stage_losses, endpoint, clean_centered = adaptive_nn_components(
        model,
        batch,
        outlier_fraction,
        outlier_scale,
    )
    base_loss = stage_losses.sum()
    core = clean_centered[:, : endpoint.shape[1], :]
    if dcd_weight > 0.0:
        dcd = density_aware_chamfer(endpoint, core, dcd_alpha, dcd_n_lambda)
        total = base_loss + dcd_weight * (dcd / dcd_alpha)
    else:
        dcd = jt.zeros((1,), dtype="float32")
        total = base_loss
    return total, stage_losses, dcd
