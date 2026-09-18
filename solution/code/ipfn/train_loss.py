"""Paper-style joint IterativePFN objective implemented entirely in Jittor."""

from __future__ import annotations

import jittor as jt


def pairwise_sq(x, y):
    return (
        (x * x).sum(2, keepdims=True)
        + (y * y).sum(2, keepdims=True).transpose(0, 2, 1)
        - 2.0 * jt.nn.bmm(x, y.transpose(0, 2, 1))
    )


def nearest_dist_and_point(x, y):
    # Built-in fused xyz KNN gives the discrete correspondence efficiently.
    # Recompute squared distance after gathering so gradients still flow to x.
    _, index = jt.misc.knn(x, y, 1)
    index = index[:, :, 0]
    point = y.reindex(
        [x.shape[0], x.shape[1], y.shape[2]],
        ["i0", "@e0(i0,i1)", "i2"], extras=[index],
    )
    return ((x - point) ** 2).sum(2), point


def adaptive_target(clean, scale, kinds, transforms):
    batch, count, _ = clean.shape
    pieces = []
    for index, kind in enumerate(kinds):
        if kind in ("gaussian", "anisotropic_gaussian"):
            base = jt.randn((count, 3)) @ transforms[index]
        elif kind == "laplace":
            uniform = jt.clamp(jt.rand((count, 3)), 1e-6, 1 - 1e-6) - 0.5
            sign = jt.where(uniform >= 0, jt.ones_like(uniform), -jt.ones_like(uniform))
            base = -sign * jt.log(1 - 2 * jt.abs(uniform))
            base = base @ transforms[index]
        elif kind == "uniform_ball":
            direction = jt.randn((count, 3))
            direction = direction / jt.sqrt((direction * direction).sum(1, keepdims=True) + 1e-12)
            base = direction * jt.rand((count, 1)).pow(1.0 / 3.0)
        else:
            raise ValueError(kind)
        pieces.append(clean[index:index + 1] + base.unsqueeze(0) * scale[index])
    return jt.concat(pieces, 0)


def joint_metric_loss(model, noisy, clean, seeds, scale, kinds, transforms,
                      noise_decay=4.0, metric_weight=0.0, fixed_targets=None):
    noisy_centered = noisy - seeds
    clean_centered = clean - seeds
    seed_dist = (noisy_centered * noisy_centered).sum(2)
    radius = seed_dist.max(1, keepdims=True)
    weights = jt.exp(-9.0 * seed_dist / (radius + 1e-12))
    weights = weights / (weights.sum(1, keepdims=True) + 1e-12)

    current = noisy_centered
    current_scale = scale
    legacy = jt.float32(0.0)
    stages = []
    for index, module in enumerate(model.feature_nets):
        displacement = module(current)
        prediction = current + displacement
        if index < model.num_modules - 1:
            current_scale = current_scale / noise_decay
            target = fixed_targets[index] if fixed_targets is not None else adaptive_target(
                clean_centered, current_scale, kinds, transforms
            )
        else:
            target = clean_centered
        _, nearest = nearest_dist_and_point(current, target)
        target_displacement = nearest - current
        legacy = legacy + (weights * ((displacement - target_displacement) ** 2).sum(2)).sum(1).mean()
        current = prediction
        stages.append(prediction)

    forward_pred = nearest_dist_and_point(current, clean_centered)[0].mean(1)
    forward_noisy = nearest_dist_and_point(noisy_centered, clean_centered)[0].mean(1)
    reverse_pred_all = nearest_dist_and_point(clean_centered, current)[0]
    reverse_noisy_all = nearest_dist_and_point(clean_centered, noisy_centered)[0]
    support = ((clean_centered * clean_centered).sum(2) <= radius).float32()
    support_count = support.sum(1) + 1e-12
    reverse_pred = (reverse_pred_all * support).sum(1) / support_count
    reverse_noisy = (reverse_noisy_all * support).sum(1) / support_count
    cd_ratio = (forward_pred + reverse_pred) / (forward_noisy + reverse_noisy + 1e-12)
    p2s_ratio = forward_pred / (forward_noisy + 1e-12)
    metric = (0.5 * cd_ratio + 0.5 * p2s_ratio).mean()
    # stop_grad() is in-place in Jittor. Detach clones, otherwise this scale
    # calculation would silently detach the actual legacy/metric objectives.
    legacy_value = legacy.clone().stop_grad()
    metric_value = metric.clone().stop_grad()
    metric_scale = legacy_value / (metric_value + 1e-12)
    total = legacy + float(metric_weight) * metric_scale * metric
    return total, legacy, metric, stages
