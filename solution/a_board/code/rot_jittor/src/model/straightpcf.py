"""Faithful Jittor port of StraightPCF (CVPR 2024).

Three training stages, selected by model_config['stage']:
  - 'vm'   : a single VelocityNet (denoising-score-matching of the flow direction).
  - 'cvm'  : `num_modules` coupled VelocityNets (each initialised from the VM
             checkpoint) trained with a direction loss + inter-module consistency.
  - 'spcf' : the coupled VelocityNets (loaded from the CVM checkpoint) plus a
             DistanceModule (encoder + scalar decoder) that scales the straight
             trajectory; velocity nets are kept in eval() mode while the distance
             module is trained with a ratio-regression + fine-tune loss.

Inference (predict): straight-flow denoising -- `tot_its` iterations, each pass
applies all coupled modules, scaled by the predicted distance scalar.

Reuses FeatureExtraction (EdgeConv) + Decoder from feature.py and the
patch_based_denoise stitching from vm.py. No torch / pytorch3d / torch_geometric.
"""
from typing import Dict, List

import jittor as jt
import numpy as np
from jittor import nn

from .feature import FeatureExtraction, Decoder, GraphConvDecoder
from .spec import ModelSpec
from .vm import patch_based_denoise
from ..data.asset import Asset


def get_random_indices(n, m):
    assert m < n
    return np.random.permutation(n)[:m]


def _make_decoder(decoder_type, z_dim, out_dim, hidden_size, scalar_reduce=True):
    """HybridPF dynamic graph-conv decoder vs the plain MLP decoder."""
    if decoder_type == 'graph':
        return GraphConvDecoder(
            z_dim=z_dim,
            dim=3,
            out_dim=out_dim,
            hidden_size=hidden_size,
            scalar_reduce=scalar_reduce,
        )
    return Decoder(
        z_dim=z_dim,
        dim=3,
        out_dim=out_dim,
        hidden_size=hidden_size,
        scalar_reduce=scalar_reduce,
    )


class VelocityNet(nn.Module):
    """Encoder (EdgeConv) + Decoder predicting a 3D flow direction."""

    def __init__(
        self,
        frame_knn,
        feat_embedding_dim,
        decoder_hidden_dim,
        decoder_type='mlp',
        attention=False,
        multiscale=False,
        film=False,
        condition_dim=0,
        encoder_type='edgeconv',
        hierarchy_hidden_dim=64,
        hierarchy_stride=4,
        hierarchy_k=16,
    ):
        super().__init__()
        self.condition_dim = int(condition_dim or 0)
        self.encoder = FeatureExtraction(
            k=frame_knn,
            input_dim=3,
            embedding_dim=feat_embedding_dim,
            attention=attention,
            multiscale=multiscale,
            film=film,
            condition_dim=self.condition_dim,
            encoder_type=encoder_type,
            hierarchy_hidden_dim=hierarchy_hidden_dim,
            hierarchy_stride=hierarchy_stride,
            hierarchy_k=hierarchy_k,
        )
        self.decoder = _make_decoder(decoder_type, self.encoder.embedding_dim, 3, decoder_hidden_dim)

    def execute(self, x, condition=None):
        B, N, d = x.shape
        feat = self.encoder(x, condition=condition)
        F_dim = feat.shape[2]
        # pass B,N so the graph decoder can rebuild the (B,N,F) structure (MLP ignores them)
        return self.decoder(c=feat.reshape(-1, F_dim), B=B, N=N).reshape(B, N, d)

    def execute_with_features(self, x, condition=None):
        """Return the frozen velocity prediction and its current point features.

        The ordinary execute path above is intentionally left unchanged so every
        existing checkpoint/config keeps its exact inference graph.  AST-001 uses
        this explicit path only when its default-off stage gate is enabled.
        """
        B, N, d = x.shape
        feat = self.encoder(x, condition=condition)
        F_dim = feat.shape[2]
        pred = self.decoder(
            c=feat.reshape(-1, F_dim), B=B, N=N
        ).reshape(B, N, d)
        return pred, feat


class StraightPCFModule(ModelSpec):

    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)
        cfg = self.model_config
        self.stage = cfg.get('stage', 'vm')
        assert self.stage in ('vm', 'cvm', 'spcf'), self.stage
        self.frame_knn = cfg['frame_knn']
        self.num_train_points = cfg['num_train_points']
        self.dsm_sigma = cfg['dsm_sigma']
        self.tot_its = cfg.get('tot_its', 2)
        # inference-time knobs (no retraining): distance scale, cloud multi-pass, TTA
        self.predict_alpha = float(cfg.get('predict_alpha', 1.0))
        self.predict_passes = int(cfg.get('predict_passes', 1))
        self.predict_tta = int(cfg.get('predict_tta', 0))
        self.predict_fusion = bool(cfg.get('predict_fusion', False))   # index-based overlap fusion
        edim = cfg['feat_embedding_dim']
        hdim = cfg['decoder_hidden_dim']

        # T1.1 CD/P2S-aligned auxiliary losses (default 0 = off -> identical to before)
        self.lam_dir = float(cfg.get('lam_dir', 0.0))    # velocity direction (cosine)
        self.lam_mag = float(cfg.get('lam_mag', 0.0))    # velocity magnitude (SmoothL1)
        self.lam_edge = float(cfg.get('lam_edge', 0.0))  # endpoint edge-vector preservation (-> CD)
        self.edge_k = int(cfg.get('edge_k', 8))
        self.lam_spacing = float(cfg.get('lam_spacing', 0.0))
        self.spacing_k = int(cfg.get('spacing_k', 8))
        self.spacing_points = int(cfg.get('spacing_points', 256))
        assert self.lam_spacing >= 0.0, self.lam_spacing
        assert self.spacing_k >= 1, self.spacing_k
        assert self.spacing_points >= 2, self.spacing_points
        self.decoder_type = cfg.get('decoder_type', 'mlp')  # 'mlp' | 'graph' (HybridPF)
        self.attention = cfg.get('attention', False)         # self-attention in encoder
        self.multiscale = cfg.get('multiscale', False)      # C3 multi-scale encoder
        self.distance_multiscale = cfg.get('distance_multiscale', self.multiscale)
        self.film = cfg.get('film', False)                  # C4 FiLM conditioning
        self.encoder_type = cfg.get('encoder_type', 'edgeconv')
        assert self.encoder_type in ('edgeconv', 'pointnext_lite'), self.encoder_type
        self.hierarchy_hidden_dim = int(cfg.get('hierarchy_hidden_dim', 64))
        self.hierarchy_stride = int(cfg.get('hierarchy_stride', 4))
        self.hierarchy_k = int(cfg.get('hierarchy_k', 16))
        self.cvm_deep_sup = cfg.get('cvm_deep_sup', False)   # C1: deep-supervise final CVM waypoint->clean
        self.cvm_dir_target = cfg.get('cvm_dir_target', 'full_residual')
        assert self.cvm_dir_target in ('full_residual', 'stage_velocity', 'blend'), self.cvm_dir_target
        self.cvm_condition = cfg.get('cvm_condition', 'none')
        assert self.cvm_condition in ('none', 'time_stage'), self.cvm_condition
        self.velocity_condition_dim = 2 if self.cvm_condition == 'time_stage' else 0
        # GFT-001: when adapting only the SPCF distance calibration to a new
        # noise distribution, preserve the proven ROT velocity field exactly.
        # Default-off keeps every existing model/config behavior unchanged.
        self.spcf_freeze_velocity = bool(
            cfg.get('spcf_freeze_velocity', False)
        )
        if self.spcf_freeze_velocity:
            assert self.stage == 'spcf', self.stage
        if self.cvm_dir_target == 'blend':
            self.cvm_stage_velocity_weight = float(cfg.get('cvm_stage_velocity_weight', 0.5))
            assert 0.0 <= self.cvm_stage_velocity_weight <= 1.0, self.cvm_stage_velocity_weight
        else:
            self.cvm_stage_velocity_weight = 1.0 if self.cvm_dir_target == 'stage_velocity' else 0.0
        self.pointwise_distance = bool(cfg.get('pointwise_distance', False))
        self.pointwise_distance_mode = cfg.get('pointwise_distance_mode', 'free')
        if self.pointwise_distance_mode == 'residual':
            self.pointwise_distance = True
        assert self.pointwise_distance_mode in ('free', 'residual'), self.pointwise_distance_mode
        self.pointwise_distance_alpha_max = float(cfg.get('pointwise_distance_alpha_max', 1.0))
        self.pointwise_distance_residual_shrink = float(cfg.get('pointwise_distance_residual_shrink', 0.25))
        self.pointwise_distance_patch_loss_weight = float(cfg.get('pointwise_distance_patch_loss_weight', 1.0))
        self.pointwise_distance_smooth_loss_weight = float(cfg.get('pointwise_distance_smooth_loss_weight', 0.0))
        self.pointwise_distance_smooth_k = int(cfg.get('pointwise_distance_smooth_k', 8))
        # UNC-001: preserve the proven patch-level distance mean and learn only
        # its input-dependent residual variance.  This is deliberately separate
        # from the rejected pointwise distance gate: the uncertainty head never
        # predicts a per-point displacement multiplier.
        self.distance_uncertainty = bool(cfg.get('distance_uncertainty', False))
        self.distance_uncertainty_gamma = float(cfg.get('distance_uncertainty_gamma', 0.0))
        self.distance_uncertainty_log_var_min = float(
            cfg.get('distance_uncertainty_log_var_min', -12.0)
        )
        self.distance_uncertainty_log_var_max = float(
            cfg.get('distance_uncertainty_log_var_max', -2.0)
        )
        self.distance_uncertainty_loss_weight = float(
            cfg.get('distance_uncertainty_loss_weight', 0.10)
        )
        self.distance_uncertainty_max_distance = float(
            cfg.get('distance_uncertainty_max_distance', 1.25)
        )
        self.distance_uncertainty_freeze_base = bool(
            cfg.get('distance_uncertainty_freeze_base', False)
        )
        if self.distance_uncertainty:
            assert self.stage == 'spcf', self.stage
            assert not self.pointwise_distance
            assert np.isfinite(self.distance_uncertainty_gamma)
            assert np.isfinite(self.distance_uncertainty_log_var_min)
            assert np.isfinite(self.distance_uncertainty_log_var_max)
            assert (
                self.distance_uncertainty_log_var_min
                < self.distance_uncertainty_log_var_max
            )
            assert np.isfinite(self.distance_uncertainty_loss_weight)
            assert self.distance_uncertainty_loss_weight > 0.0
            assert np.isfinite(self.distance_uncertainty_max_distance)
            assert self.distance_uncertainty_max_distance > 0.0
        # NAS-001: AID (arXiv:2509.14560) motivates estimating the input noise
        # state before selecting a denoising schedule.  Noise2Score3D
        # (arXiv:2503.09283) likewise makes the denoising magnitude explicitly
        # sigma-dependent and trains across a continuous Gaussian-noise range.
        # Keep this head causally isolated from ROT: only the new scalar decoder
        # is trainable, and max_shrink=0 bypasses it exactly during inference.
        self.distance_noise_adaptive = bool(
            cfg.get('distance_noise_adaptive', False)
        )
        self.distance_noise_min = float(cfg.get('distance_noise_min', 0.006))
        self.distance_noise_max = float(cfg.get('distance_noise_max', 0.016))
        self.distance_noise_schedule_low = float(
            cfg.get('distance_noise_schedule_low', 0.008)
        )
        self.distance_noise_schedule_high = float(
            cfg.get('distance_noise_schedule_high', 0.014)
        )
        self.distance_noise_max_shrink = float(
            cfg.get('distance_noise_max_shrink', 0.0)
        )
        self.distance_noise_loss_weight = float(
            cfg.get('distance_noise_loss_weight', 1.0)
        )
        self.distance_noise_freeze_base = bool(
            cfg.get('distance_noise_freeze_base', False)
        )
        self.distance_noise_batch_mean = bool(
            cfg.get('distance_noise_batch_mean', True)
        )
        self.distance_noise_decoder_type = cfg.get(
            'distance_noise_decoder_type', 'mlp'
        )
        if self.distance_noise_adaptive:
            assert self.stage == 'spcf', self.stage
            assert not self.pointwise_distance
            assert not self.distance_uncertainty
            assert self.distance_noise_freeze_base
            assert self.distance_noise_decoder_type in ('mlp', 'graph')
            assert np.isfinite(self.distance_noise_min)
            assert np.isfinite(self.distance_noise_max)
            assert self.distance_noise_min < self.distance_noise_max
            assert (
                self.distance_noise_min
                <= self.distance_noise_schedule_low
                < self.distance_noise_schedule_high
                <= self.distance_noise_max
            )
            assert np.isfinite(self.distance_noise_max_shrink)
            assert self.distance_noise_max_shrink >= 0.0
            assert np.isfinite(self.distance_noise_loss_weight)
            assert self.distance_noise_loss_weight > 0.0
        # AST-001: a differentiable, stage-wise relaxation of ASDN's adaptive
        # stopping.  Unlike the rejected static pointwise distance gate, this
        # head is recomputed from the current velocity representation at every
        # internal update.  strength=0 is the exact matched-constant control.
        self.adaptive_stage_stop = bool(
            cfg.get('adaptive_stage_stop', False)
        )
        self.adaptive_stage_stop_strength = float(
            cfg.get('adaptive_stage_stop_strength', 0.0)
        )
        self.adaptive_stage_stop_base_shrink = float(
            cfg.get('adaptive_stage_stop_base_shrink', 0.0)
        )
        self.adaptive_stage_stop_loss_weight = float(
            cfg.get('adaptive_stage_stop_loss_weight', 1.0)
        )
        self.adaptive_stage_stop_initial_gate = float(
            cfg.get('adaptive_stage_stop_initial_gate', 0.99)
        )
        self.adaptive_stage_stop_decoder_type = cfg.get(
            'adaptive_stage_stop_decoder_type', 'mlp'
        )
        self.adaptive_stage_stop_freeze_base = bool(
            cfg.get('adaptive_stage_stop_freeze_base', False)
        )
        if self.adaptive_stage_stop:
            assert self.stage == 'spcf', self.stage
            assert not self.pointwise_distance
            assert not self.distance_uncertainty
            assert not self.distance_noise_adaptive
            assert self.adaptive_stage_stop_freeze_base
            assert self.adaptive_stage_stop_decoder_type in ('mlp', 'graph')
            assert 0.0 <= self.adaptive_stage_stop_strength <= 1.0
            assert np.isfinite(self.adaptive_stage_stop_base_shrink)
            assert self.adaptive_stage_stop_base_shrink >= 0.0
            assert np.isfinite(self.adaptive_stage_stop_loss_weight)
            assert self.adaptive_stage_stop_loss_weight > 0.0
            assert 0.0 < self.adaptive_stage_stop_initial_gate < 1.0
        self.num_modules = 1 if self.stage == 'vm' else cfg.get('num_modules', 2)
        self.velocity_nets = nn.ModuleList(
            [VelocityNet(
                self.frame_knn,
                edim,
                hdim,
                decoder_type=self.decoder_type,
                attention=self.attention,
                multiscale=self.multiscale,
                film=self.film,
                condition_dim=self.velocity_condition_dim,
                encoder_type=self.encoder_type,
                hierarchy_hidden_dim=self.hierarchy_hidden_dim,
                hierarchy_stride=self.hierarchy_stride,
                hierarchy_k=self.hierarchy_k,
             )
             for _ in range(self.num_modules)]
        )
        if self.spcf_freeze_velocity:
            self._stop_module_grad(self.velocity_nets)

        if self.stage == 'spcf':
            # DistanceModule: predicts a scalar in (0,1) scaling the trajectory.
            self.encoder = FeatureExtraction(
                k=self.frame_knn, input_dim=3, embedding_dim=edim,
                distance_estimation=cfg.get('distance_estimation', True),
                attention=self.attention,
                multiscale=self.distance_multiscale,
                film=self.film,
                encoder_type=self.encoder_type,
                hierarchy_hidden_dim=self.hierarchy_hidden_dim,
                hierarchy_stride=self.hierarchy_stride,
                hierarchy_k=self.hierarchy_k,
            )
            if self.pointwise_distance and self.pointwise_distance_mode == 'residual':
                self.patch_decoder = _make_decoder(
                    self.decoder_type,
                    edim,
                    1,
                    hdim,
                    scalar_reduce=True,
                )
                self.decoder = _make_decoder(
                    self.decoder_type,
                    edim,
                    1,
                    hdim,
                    scalar_reduce=False,
                )
            else:
                self.decoder = _make_decoder(
                    self.decoder_type,
                    edim,
                    1,
                    hdim,
                    scalar_reduce=not self.pointwise_distance,
                )
            if self.distance_uncertainty:
                self.uncertainty_decoder = _make_decoder(
                    self.decoder_type,
                    edim,
                    1,
                    hdim,
                    scalar_reduce=True,
                )
                self._zero_uncertainty_output()
                if self.distance_uncertainty_freeze_base:
                    self._freeze_uncertainty_base()
            if self.distance_noise_adaptive:
                self.noise_decoder = _make_decoder(
                    self.distance_noise_decoder_type,
                    edim,
                    1,
                    hdim,
                    scalar_reduce=True,
                )
                self._zero_noise_output()
                self._freeze_noise_base()
            if self.adaptive_stage_stop:
                # One lightweight point head per coupled velocity stage.  The
                # final channel is the normalized remaining-step fraction.
                self.stage_stop_decoders = nn.ModuleList([
                    _make_decoder(
                        self.adaptive_stage_stop_decoder_type,
                        edim + 1,
                        1,
                        hdim,
                        scalar_reduce=False,
                    )
                    for _ in range(self.num_modules)
                ])
                self._initialize_stage_stop_outputs()
                self._freeze_adaptive_stage_stop_base()

    def _initialize_stage_stop_outputs(self):
        """Conservatively initialize every new gate close to 'continue'."""
        bias = float(np.log(
            self.adaptive_stage_stop_initial_gate
            / (1.0 - self.adaptive_stage_stop_initial_gate)
        ))
        for decoder in self.stage_stop_decoders:
            if hasattr(decoder, 'lin_out'):
                layer = decoder.lin_out
            else:
                layer = decoder.lin_3
            layer.weight.assign(jt.zeros_like(layer.weight))
            layer.bias.assign(jt.ones_like(layer.bias) * bias)

    def _zero_uncertainty_output(self):
        """Start at the midpoint of the configured log-variance interval."""
        if hasattr(self.uncertainty_decoder, 'lin_out'):
            layer = self.uncertainty_decoder.lin_out
        else:
            layer = self.uncertainty_decoder.lin_3
        layer.weight.assign(jt.zeros_like(layer.weight))
        layer.bias.assign(jt.zeros_like(layer.bias))

    @staticmethod
    def _stop_module_grad(module):
        for parameter in module.parameters():
            parameter.stop_grad()

    def _freeze_uncertainty_base(self):
        """Train only the new variance head while preserving the ROT function."""
        self._stop_module_grad(self.velocity_nets)
        self._stop_module_grad(self.encoder)
        self._stop_module_grad(self.decoder)

    def _zero_noise_output(self):
        """Initialize the predicted noise level at the configured midpoint."""
        if hasattr(self.noise_decoder, 'lin_out'):
            layer = self.noise_decoder.lin_out
        else:
            layer = self.noise_decoder.lin_3
        layer.weight.assign(jt.zeros_like(layer.weight))
        layer.bias.assign(jt.zeros_like(layer.bias))

    def _freeze_noise_base(self):
        """Train only the supervised noise head; preserve every ROT tensor."""
        self._stop_module_grad(self.velocity_nets)
        self._stop_module_grad(self.encoder)
        self._stop_module_grad(self.decoder)

    def _freeze_adaptive_stage_stop_base(self):
        self._stop_module_grad(self.velocity_nets)
        self._stop_module_grad(self.encoder)
        self._stop_module_grad(self.decoder)

    @staticmethod
    def _apply_fixed_distance_shrink(distance, shrink):
        if shrink == 0.0:
            return distance
        return jt.maximum(
            distance - shrink,
            jt.zeros_like(distance),
        )

    @staticmethod
    def _adaptive_stage_alpha_target(current, clean, proposed_step):
        """One-step optimal coefficient along the frozen proposed update."""
        residual = clean - current
        target = (residual * proposed_step).sum(dim=-1, keepdims=True)
        target = target / (
            (proposed_step ** 2).sum(dim=-1, keepdims=True) + 1e-8
        )
        target = jt.maximum(target, jt.zeros_like(target))
        return jt.minimum(target, jt.ones_like(target))

    def _adaptive_stage_gate(self, features, mod, remaining_fraction):
        B, N, _ = features.shape
        remaining = (
            jt.ones((B, N, 1)) * float(remaining_fraction)
        )
        conditioned = jt.concat([features, remaining], dim=-1)
        F_dim = conditioned.shape[2]
        return self.stage_stop_decoders[mod](
            c=conditioned.reshape(-1, F_dim), B=B, N=N
        ).reshape(B, N, 1)

    def _distance_noise_level(self, features, B, N):
        unit = self.noise_decoder(
            c=features.reshape(-1, features.shape[2]),
            B=B,
            N=N,
        ).reshape(B)
        span = self.distance_noise_max - self.distance_noise_min
        return self.distance_noise_min + span * unit

    def _distance_noise_regression_loss(self, prediction, target):
        """Regress normalized sigma so the loss scale is range-independent."""
        span = self.distance_noise_max - self.distance_noise_min
        target_unit = (target - self.distance_noise_min) / span
        target_unit = jt.maximum(target_unit, jt.zeros_like(target_unit))
        target_unit = jt.minimum(target_unit, jt.ones_like(target_unit))
        prediction_unit = (prediction - self.distance_noise_min) / span
        return ((prediction_unit - target_unit.detach()) ** 2).mean()

    @staticmethod
    def _apply_distance_noise_schedule(
        distance,
        predicted_noise,
        schedule_low,
        schedule_high,
        maximum_shrink,
        batch_mean,
    ):
        """Linearly taper the validated shrink to zero as noise increases."""
        if batch_mean:
            predicted_noise = (
                jt.ones_like(predicted_noise) * predicted_noise.mean()
            )
        shrink_unit = (schedule_high - predicted_noise) / (
            schedule_high - schedule_low
        )
        shrink_unit = jt.maximum(shrink_unit, jt.zeros_like(shrink_unit))
        shrink_unit = jt.minimum(shrink_unit, jt.ones_like(shrink_unit))
        adjusted = distance - maximum_shrink * shrink_unit.reshape(distance.shape)
        return jt.maximum(adjusted, jt.zeros_like(adjusted))

    def _distance_log_variance(self, features, B, N):
        unit = self.uncertainty_decoder(
            c=features.reshape(-1, features.shape[2]),
            B=B,
            N=N,
        ).reshape(B)
        span = (
            self.distance_uncertainty_log_var_max
            - self.distance_uncertainty_log_var_min
        )
        return self.distance_uncertainty_log_var_min + span * unit

    @staticmethod
    def _heteroscedastic_distance_nll(mean, target, log_variance):
        """Gaussian NLL for a frozen mean and learned input-dependent variance."""
        residual = target - mean.detach()
        return 0.5 * (
            jt.exp(-log_variance) * residual * residual + log_variance
        ).mean()

    def _distance_uncertainty_nll(self, mean, target, log_variance):
        # Inference applies predict_alpha before the uncertainty correction, so
        # the variance head must learn residuals around that exact deployed mean.
        calibrated_mean = self.predict_alpha * mean.detach()
        return self._heteroscedastic_distance_nll(
            calibrated_mean,
            target,
            log_variance,
        )

    @staticmethod
    def _apply_distance_uncertainty(
        distance,
        log_variance,
        gamma,
        maximum_distance,
    ):
        # Signed gamma lets the paired screen distinguish AID-style expansion
        # from risk-conservative contraction.  Zero bypasses this helper at the
        # call site so the baseline path remains bitwise unchanged.
        sigma = jt.exp(0.5 * log_variance).reshape(distance.shape)
        adjusted = distance + gamma * sigma
        adjusted = jt.maximum(adjusted, jt.zeros_like(adjusted))
        return jt.minimum(
            adjusted,
            jt.ones_like(adjusted) * maximum_distance,
        )

    @staticmethod
    def _combine_patch_residual_distance(patch_gate, residual_gate, residual_shrink):
        patch_gate = jt.maximum(patch_gate, jt.ones_like(patch_gate) * 1e-4)
        patch_gate = jt.minimum(patch_gate, jt.ones_like(patch_gate) * (1.0 - 1e-4))
        centered_residual = (residual_gate - 0.5) * 2.0
        patch_logit = jt.log(patch_gate / (1.0 - patch_gate))
        return jt.sigmoid(patch_logit + residual_shrink * centered_residual)

    @staticmethod
    def _gather_point_values(values, knn):
        B, n, c = values.shape
        k = knn.shape[2]
        base = (jt.arange(B) * n).reshape(B, 1, 1)
        flat = (knn + base).reshape(-1)
        return values.reshape(B * n, c)[flat].reshape(B, n, k, c)

    @staticmethod
    def _pointwise_gate_smoothness(gate, points, k):
        if k <= 0 or gate.shape[1] <= 1:
            return (gate * 0.0).mean()
        k = min(k, gate.shape[1] - 1)
        d = ((points.unsqueeze(2) - points.unsqueeze(1)) ** 2).sum(-1)
        knn = jt.argsort(d, dim=-1)[0][:, :, 1:k + 1]
        neighbor_gate = StraightPCFModule._gather_point_values(gate, knn)
        return jt.abs(gate.unsqueeze(2) - neighbor_gate).mean()

    def _make_velocity_condition(self, remaining, mod):
        if self.cvm_condition == 'none':
            return None
        if len(remaining.shape) == 3:
            remaining = remaining.reshape(remaining.shape[0], -1).mean(dim=1, keepdims=True)
        elif len(remaining.shape) == 1:
            remaining = remaining.reshape(-1, 1)
        else:
            remaining = remaining.reshape(remaining.shape[0], -1).mean(dim=1, keepdims=True)
        denom = max(1, self.num_modules - 1)
        stage = jt.ones_like(remaining) * (float(mod) / float(denom))
        return jt.concat([remaining, stage], dim=1)

    def _velocity(self, mod, points, remaining):
        condition = self._make_velocity_condition(remaining, mod)
        return self.velocity_nets[mod](points, condition=condition)

    def _velocity_with_features(self, mod, points, remaining):
        condition = self._make_velocity_condition(remaining, mod)
        return self.velocity_nets[mod].execute_with_features(
            points, condition=condition
        )

    # ----- checkpoint chaining between stages -----
    def init_from_stage(self, ckpt_path: str):
        """Load a previous-stage checkpoint into this stage's submodules.
        vm->cvm: copy the single VM weights into every coupled module.
        cvm->spcf: copy the coupled velocity_nets verbatim (distance module stays fresh).
        """
        prev = jt.load(ckpt_path)
        sd = prev['state_dict'] if isinstance(prev, dict) and 'state_dict' in prev else prev
        # Materialize independent NumPy copies. Passing checkpoint Vars directly to
        # load_state_dict aliases their autograd source in Jittor 1.3.11, which can
        # silently leave all but one broadcast CVM module without gradients.
        src_mods = {}
        for k, v in sd.items():
            if k.startswith('velocity_nets.'):
                idx = k.split('.')[1]
                value = v.numpy() if isinstance(v, jt.Var) else np.asarray(v)
                src_mods.setdefault(idx, {})[k.split('.', 2)[2]] = (
                    np.ascontiguousarray(value).copy()
                )
        if not src_mods:
            raise ValueError(f"no velocity_nets.* found in {ckpt_path}")
        src_keys = sorted(src_mods.keys(), key=int)
        for i in range(self.num_modules):
            # vm(1 module)->cvm(N): broadcast module 0 to all; cvm->spcf: 1-1 map
            src = src_mods[src_keys[i] if i < len(src_keys) else src_keys[0]]
            self.velocity_nets[i].load_parameters(
                {key: value.copy() for key, value in src.items()}
            )
        print(f"\033[92mloaded velocity_nets from {ckpt_path} "
              f"({len(src_keys)} src -> {self.num_modules} dst)\033[0m")

    # ----- training -----
    def _unpack(self, batch):
        # shapes after collate: (B, 1, M, 3) and (B, 1) -> squeeze patch dim
        pcl_clean = batch['pcl_clean'].reshape(-1, batch['pcl_clean'].shape[-2], 3)
        pcl_noisy_L2 = batch['pcl_noisy_L2'].reshape(-1, batch['pcl_noisy_L2'].shape[-2], 3)
        seeds_t = batch['seed_points_t'].reshape(-1, 1, 3)
        t = batch['original_time_step'].reshape(-1)
        return pcl_clean, pcl_noisy_L2, seeds_t, t

    def training_step(self, batch: Dict) -> Dict:
        pcl_clean, pcl_noisy_L2, seeds_t, t = self._unpack(batch)
        if self.adaptive_stage_stop:
            loss = self._loss_adaptive_stage_stop(
                pcl_clean,
                pcl_noisy_L2,
            )
        elif self.distance_noise_adaptive:
            assert 'noise_std' in batch
            loss = self._loss_distance_noise(
                pcl_noisy_L2,
                batch['noise_std'].reshape(-1),
            )
        elif self.stage == 'vm':
            loss = self._loss_vm(pcl_clean, pcl_noisy_L2, seeds_t, t)
        elif self.stage == 'cvm':
            loss = self._loss_cvm(pcl_clean, pcl_noisy_L2, seeds_t, t)
        else:
            loss = self._loss_spcf(pcl_clean, pcl_noisy_L2, seeds_t, t)
        return {"loss": loss}

    def execute(self, **kwargs) -> Dict:
        return self.training_step(**kwargs)

    def _loss_distance_noise(self, pcl_noisy, target_noise):
        """Head-only sigma regression on the exact inference-state patch."""
        self.velocity_nets.eval()
        self.encoder.eval()
        self.decoder.eval()
        self.noise_decoder.train()
        if not isinstance(pcl_noisy, jt.Var):
            pcl_noisy = jt.array(pcl_noisy)
        if not isinstance(target_noise, jt.Var):
            target_noise = jt.array(target_noise)
        B, N, _ = pcl_noisy.shape
        # cKDTree returns the queried noisy seed as the first neighbour.  Using
        # it as the origin matches patch_based_denoise's inference centering and
        # avoids teaching the head from the random interpolation timestep.
        pcl_noisy_c = pcl_noisy - pcl_noisy[:, :1, :]
        features = self.encoder(pcl_noisy_c)
        prediction = self._distance_noise_level(features, B, N)
        loss = self._distance_noise_regression_loss(prediction, target_noise)
        return self.distance_noise_loss_weight * loss

    def _loss_adaptive_stage_stop(self, pcl_clean, pcl_noisy):
        """Train only the current-state next-step utility heads.

        The frozen ROT trajectory starts from the exact fully-noisy inference
        patch.  Teacher-forced one-step optima generate later states without
        backpropagating through either clean-derived targets or the base model.
        """
        self.velocity_nets.eval()
        self.encoder.eval()
        self.decoder.eval()
        self.stage_stop_decoders.train()
        if not isinstance(pcl_clean, jt.Var):
            pcl_clean = jt.array(pcl_clean)
        if not isinstance(pcl_noisy, jt.Var):
            pcl_noisy = jt.array(pcl_noisy)
        B, N, _ = pcl_noisy.shape
        origin = pcl_noisy[:, :1, :]
        current = (pcl_noisy - origin).detach()
        clean = (pcl_clean - origin).detach()

        distance_features = self.encoder(current).detach()
        F_dim = distance_features.shape[2]
        distance = self.decoder(
            c=distance_features.reshape(-1, F_dim), B=B, N=N
        ).reshape(B, 1, 1)
        distance = (distance * self.predict_alpha).detach()
        distance = self._apply_fixed_distance_shrink(
            distance,
            self.adaptive_stage_stop_base_shrink,
        ).detach()

        total_loss = 0.0
        total_steps = self.tot_its * self.num_modules
        for it in range(self.tot_its):
            velocity_remaining = distance * (
                float(self.tot_its - it) / float(self.tot_its)
            )
            for mod in range(self.num_modules):
                step_index = it * self.num_modules + mod
                remaining_fraction = float(
                    total_steps - step_index
                ) / float(total_steps)
                direction, features = self._velocity_with_features(
                    mod,
                    current,
                    velocity_remaining,
                )
                direction = direction.detach()
                features = features.detach()
                proposed_step = (
                    (1.0 / self.tot_its)
                    * (1.0 / self.num_modules)
                    * distance
                    * direction
                ).detach()
                target = self._adaptive_stage_alpha_target(
                    current,
                    clean,
                    proposed_step,
                ).detach()
                gate = self._adaptive_stage_gate(
                    features,
                    mod,
                    remaining_fraction,
                )
                total_loss = total_loss + ((gate - target) ** 2).mean()
                current = (current + target * proposed_step).detach()
        return (
            self.adaptive_stage_stop_loss_weight
            * total_loss
            / float(total_steps)
        )

    # ----- T1.1 CD/P2S-aligned auxiliary losses -----
    @staticmethod
    def _huber(x):
        ax = jt.abs(x)
        q = (ax < 1.0).float32()
        return (q * 0.5 * x * x + (1.0 - q) * (ax - 0.5)).mean()

    def _gather_edges(self, x, knn):
        # x:(B,n,3) knn:(B,n,k) -> edge vectors (B,n,k,3) = x[j]-x[i]
        B, n, _ = x.shape
        k = knn.shape[2]
        base = (jt.arange(B) * n).reshape(B, 1, 1)
        flat = (knn + base).reshape(-1)
        nb = x.reshape(B * n, 3)[flat].reshape(B, n, k, 3)
        return nb - x.unsqueeze(2)

    def _local_spacing_loss(self, endpoint, clean):
        """Match clean-neighbour spacing without constraining edge direction.

        The clean kNN graph supplies a stable surface topology target.  Comparing
        log edge lengths is rotation/translation invariant and penalizes both
        point collapse and excessive spreading, while avoiding the directional
        over-constraint that made the earlier edge-vector auxiliary regress.
        """
        _, n, _ = clean.shape
        count = min(n, self.spacing_points)
        if count < n:
            # Patch points are radius-sorted by cKDTree.  Evenly spaced indices
            # cover the full patch deterministically instead of only its centre.
            subset = np.linspace(0, n - 1, count, dtype=np.int64)
            endpoint = endpoint[:, subset, :]
            clean = clean[:, subset, :]
        k = min(self.spacing_k, count - 1)
        distances = ((clean.unsqueeze(2) - clean.unsqueeze(1)) ** 2).sum(-1)
        knn = jt.argsort(distances, dim=-1)[0][:, :, 1:k + 1]
        endpoint_edges = self._gather_edges(endpoint, knn)
        clean_edges = self._gather_edges(clean, knn)
        endpoint_length = jt.sqrt((endpoint_edges ** 2).sum(-1) + 1e-12)
        clean_length = jt.sqrt((clean_edges ** 2).sum(-1) + 1e-12)
        log_ratio = jt.log((endpoint_length + 1e-6) / (clean_length + 1e-6))
        return self._huber(log_ratio)

    def _aux_losses(self, pred, target, endpoint, clean):
        # all (B,n,3) on the training subset
        aux = 0.0
        if self.lam_dir > 0 or self.lam_mag > 0:
            pn = jt.sqrt((pred ** 2).sum(-1) + 1e-8)
            tn = jt.sqrt((target ** 2).sum(-1) + 1e-8)
            if self.lam_dir > 0:
                cos = (pred * target).sum(-1) / (pn * tn + 1e-8)
                aux = aux + self.lam_dir * (1.0 - cos).mean()
            if self.lam_mag > 0:
                aux = aux + self.lam_mag * self._huber(pn - tn)
        if self.lam_edge > 0:
            ce, ee = clean, endpoint
            if ce.shape[1] > 256:                          # cap O(n^2) edge cost
                sub = get_random_indices(ce.shape[1], 256)
                ce, ee = ce[:, sub, :], ee[:, sub, :]
            d = ((ce.unsqueeze(2) - ce.unsqueeze(1)) ** 2).sum(-1)         # (B,n,n)
            knn = jt.argsort(d, dim=-1)[0][:, :, 1:self.edge_k + 1]        # (B,n,k) excl self
            diff = self._gather_edges(ee, knn) - self._gather_edges(ce, knn)
            aux = aux + self.lam_edge * self._huber(diff)
        return aux

    def _loss_vm(self, pcl_clean, pcl_noisy_L2, seeds_t, t):
        B, N, d = pcl_noisy_L2.shape
        tt = t.reshape(B, 1, 1)
        pcl_noisy = tt * pcl_clean + (1 - tt) * pcl_noisy_L2  # interpolated (pat_t)
        # center by the interpolated seed
        pcl_clean_c = pcl_clean - seeds_t
        pcl_noisy_L2_c = pcl_noisy_L2 - seeds_t
        pcl_noisy_c = pcl_noisy - seeds_t
        target = pcl_clean_c - pcl_noisy_L2_c  # == pat_B - pat_A
        pred = self.velocity_nets[0](pcl_noisy_c)
        idx = get_random_indices(N, self.num_train_points)
        pred_s, target_s = pred[:, idx, :], target[:, idx, :]
        loss = (((pred_s - target_s) ** 2.0) / self.dsm_sigma).sum(dim=-1).mean()
        if self.lam_dir > 0 or self.lam_mag > 0 or self.lam_edge > 0:
            # endpoint estimate at this t: x_t + (1-t)*v  ≈ clean (-> edge/P2S structure)
            rem = (1.0 - t.reshape(B, 1, 1))
            endpoint_s = pcl_noisy_c[:, idx, :] + rem * pred_s
            loss = loss + self._aux_losses(pred_s, target_s, endpoint_s, pcl_clean_c[:, idx, :])
        return loss

    def _cvm_direction_target(self, pcl_clean, pcl_noisy_L2, seeds_t, t, mod, current):
        B = pcl_noisy_L2.shape[0]
        cp1 = (t * (self.num_modules - (mod + 1)) + (mod + 1)) / self.num_modules
        cp1 = cp1.reshape(B, 1, 1)
        next_waypoint = cp1 * pcl_clean + (1 - cp1) * pcl_noisy_L2 - seeds_t
        full_target = pcl_clean - pcl_noisy_L2
        step_scale = (1.0 - t.reshape(B, 1, 1)) / self.num_modules
        stage_target = (next_waypoint - current.detach()) / (step_scale + 1e-8)
        w = self.cvm_stage_velocity_weight
        target = (1.0 - w) * full_target + w * stage_target
        return target, next_waypoint

    def _loss_cvm(self, pcl_clean, pcl_noisy_L2, seeds_t, t):
        B, N, d = pcl_noisy_L2.shape
        total_dir = 0.0
        total_cons = 0.0
        curr = (t * self.num_modules) / self.num_modules
        curr = curr.reshape(B, 1, 1)
        pcl_noisy = curr * pcl_clean + (1 - curr) * pcl_noisy_L2
        pcl_noisy = pcl_noisy - seeds_t
        remaining = 1.0 - t.reshape(B, 1)
        for mod in range(self.num_modules):
            dir_target, next_waypoint = self._cvm_direction_target(
                pcl_clean,
                pcl_noisy_L2,
                seeds_t,
                t,
                mod,
                pcl_noisy,
            )
            pred_dir = self._velocity(mod, pcl_noisy, remaining)
            total_dir = total_dir + ((pred_dir - dir_target) ** 2).sum(dim=-1).mean()
            pcl_noisy = pcl_noisy + ((1.0 - t.reshape(B, 1, 1)) / self.num_modules) * pred_dir
            if mod < self.num_modules - 1:
                total_cons = total_cons + ((next_waypoint - pcl_noisy) ** 2).sum(dim=-1).mean()
            elif self.cvm_deep_sup:
                total_cons = total_cons + ((next_waypoint - pcl_noisy) ** 2).sum(dim=-1).mean()
        loss = (total_dir + 10 * total_cons) / self.dsm_sigma
        if self.lam_dir > 0 or self.lam_mag > 0 or self.lam_edge > 0:
            # pcl_noisy is now the final endpoint (~ clean_c); last pred_dir is the velocity
            loss = loss + self._aux_losses(pred_dir, pcl_clean - pcl_noisy_L2, pcl_noisy, pcl_clean - seeds_t)
        if self.lam_spacing > 0:
            loss = loss + self.lam_spacing * self._local_spacing_loss(
                pcl_noisy,
                pcl_clean - seeds_t,
            )
        return loss

    def _loss_spcf(self, pcl_clean, pcl_noisy_L2, seeds_t, t):
        # velocity nets frozen-ish (eval mode); train the distance module
        self.velocity_nets.eval()
        if self.distance_uncertainty and self.distance_uncertainty_freeze_base:
            # The outer trainer calls model.train() at every epoch.  Re-assert
            # eval mode here so frozen ROT BatchNorm/dropout state is immutable,
            # while the independent uncertainty decoder remains in train mode.
            self.encoder.eval()
            self.decoder.eval()
        B, N, d = pcl_noisy_L2.shape
        tt = t.reshape(B, 1, 1)
        pcl_noisy = tt * pcl_clean + (1 - tt) * pcl_noisy_L2
        num = jt.sqrt(((pcl_clean - pcl_noisy) ** 2).sum(dim=-1))
        den = jt.sqrt(((pcl_clean - pcl_noisy_L2) ** 2).sum(dim=-1))
        ratio = num[:, 0] / (den[:, 0] + 1e-12)

        pcl_clean_c = pcl_clean - seeds_t
        pcl_noisy_c = pcl_noisy - seeds_t

        feat_d = self.encoder(pcl_noisy_c)
        F_d = feat_d.shape[2]
        uncertainty_loss = None
        if self.pointwise_distance:
            ratio_remaining = ratio.reshape(B, 1)
            if self.pointwise_distance_mode == 'residual':
                patch_d = self.patch_decoder(c=feat_d.reshape(-1, F_d), B=B, N=N).reshape(B, 1, 1)
                residual_d = self.decoder(c=feat_d.reshape(-1, F_d), B=B, N=N).reshape(B, N, 1)
                pred_d = self._combine_patch_residual_distance(
                    patch_d,
                    residual_d,
                    self.pointwise_distance_residual_shrink,
                )
                patch_loss = ((patch_d.reshape(B) - ratio) ** 2).mean()
            else:
                pred_d = self.decoder(c=feat_d.reshape(-1, F_d), B=B, N=N).reshape(B, N, 1)
                patch_loss = 0.0
            probe = pcl_noisy_c
            total_step = jt.zeros_like(pcl_noisy_c)
            for mod in range(self.num_modules):
                pred_dir_probe = self._velocity(mod, probe, ratio_remaining)
                step = (1.0 / self.num_modules) * pred_dir_probe
                total_step = total_step + step
                probe = probe + step
            step_sg = total_step.detach()
            residual = pcl_clean_c - pcl_noisy_c
            alpha_target = (residual * step_sg).sum(dim=-1, keepdims=True)
            alpha_target = alpha_target / ((step_sg ** 2).sum(dim=-1, keepdims=True) + 1e-8)
            alpha_target = jt.maximum(alpha_target, jt.zeros_like(alpha_target))
            alpha_target = jt.minimum(
                alpha_target,
                jt.ones_like(alpha_target) * self.pointwise_distance_alpha_max,
            )
            loss = ((pred_d - alpha_target) ** 2).mean()
            if self.pointwise_distance_mode == 'residual':
                loss = loss + self.pointwise_distance_patch_loss_weight * patch_loss
            if self.pointwise_distance_smooth_loss_weight > 0:
                smooth = self._pointwise_gate_smoothness(
                    pred_d,
                    pcl_noisy_c,
                    self.pointwise_distance_smooth_k,
                )
                loss = loss + self.pointwise_distance_smooth_loss_weight * smooth
        else:
            pred_d = self.decoder(c=feat_d.reshape(-1, F_d), B=B, N=N)
            pred_d = pred_d.reshape(B)
            loss = ((pred_d - ratio) ** 2).mean()
            if self.distance_uncertainty:
                log_variance = self._distance_log_variance(feat_d, B, N)
                uncertainty_loss = self._distance_uncertainty_nll(
                    pred_d,
                    ratio,
                    log_variance,
                )
            pred_d = pred_d.reshape(B, 1, 1)

        cur = pcl_noisy_c
        velocity_remaining = pred_d.detach()
        for mod in range(self.num_modules):
            pred_dir = self._velocity(mod, cur, velocity_remaining)
            cur = cur + (1.0 / self.num_modules) * pred_d * pred_dir
        finetune = 2e2 * ((pcl_clean_c - cur) ** 2).sum(dim=-1).mean()
        spcf_loss = (loss + finetune) / self.dsm_sigma
        if uncertainty_loss is not None:
            spcf_loss = (
                spcf_loss
                + self.distance_uncertainty_loss_weight * uncertainty_loss
            )
        if self.lam_dir > 0 or self.lam_mag > 0 or self.lam_edge > 0:
            # cur is the final distance-scaled endpoint (~ clean_c)
            spcf_loss = spcf_loss + self._aux_losses(pred_dir, pcl_clean_c - pcl_noisy_c, cur, pcl_clean_c)
        if self.lam_spacing > 0:
            spcf_loss = spcf_loss + self.lam_spacing * self._local_spacing_loss(
                cur,
                pcl_clean_c,
            )
        return spcf_loss

    # ----- inference (called per-patch by patch_based_denoise) -----
    def denoise_langevin_dynamics(self, pcl_noisy, num_steps: int = None):
        B, N, d = pcl_noisy.shape
        with jt.no_grad():
            self.eval()
            pcl_next = pcl_noisy.clone()
            log_variance = None
            predicted_noise = None
            if self.stage == 'spcf':
                feat_d = self.encoder(pcl_next)
                F_d = feat_d.shape[2]
                if self.pointwise_distance and self.pointwise_distance_mode == 'residual':
                    patch_d = self.patch_decoder(c=feat_d.reshape(-1, F_d), B=B, N=N).reshape(B, 1, 1)
                    residual_d = self.decoder(c=feat_d.reshape(-1, F_d), B=B, N=N).reshape(B, N, 1)
                    pred_d = self._combine_patch_residual_distance(
                        patch_d,
                        residual_d,
                        self.pointwise_distance_residual_shrink,
                    )
                elif self.pointwise_distance:
                    pred_d = self.decoder(c=feat_d.reshape(-1, F_d), B=B, N=N).reshape(B, N, 1)
                else:
                    pred_d = self.decoder(c=feat_d.reshape(-1, F_d), B=B, N=N).reshape(B, 1, 1)
                    if (
                        self.distance_uncertainty
                        and self.distance_uncertainty_gamma != 0.0
                    ):
                        log_variance = self._distance_log_variance(feat_d, B, N)
                    if (
                        self.distance_noise_adaptive
                        and self.distance_noise_max_shrink != 0.0
                    ):
                        predicted_noise = self._distance_noise_level(feat_d, B, N)
            else:
                pred_d = jt.ones((B, 1, 1))
            pred_d = pred_d * self.predict_alpha   # inference distance scale
            if log_variance is not None:
                pred_d = self._apply_distance_uncertainty(
                    pred_d,
                    log_variance,
                    self.distance_uncertainty_gamma,
                    self.distance_uncertainty_max_distance,
                )
            if predicted_noise is not None:
                pred_d = self._apply_distance_noise_schedule(
                    pred_d,
                    predicted_noise,
                    self.distance_noise_schedule_low,
                    self.distance_noise_schedule_high,
                    self.distance_noise_max_shrink,
                    self.distance_noise_batch_mean,
                )
            if self.adaptive_stage_stop:
                pred_d = self._apply_fixed_distance_shrink(
                    pred_d,
                    self.adaptive_stage_stop_base_shrink,
                )
            its = self.tot_its
            total_steps = its * self.num_modules
            for it in range(its):
                velocity_remaining = pred_d * (float(its - it) / float(its))
                for mod in range(self.num_modules):
                    gate = None
                    if (
                        self.adaptive_stage_stop
                        and self.adaptive_stage_stop_strength != 0.0
                    ):
                        pred_dir, features = self._velocity_with_features(
                            mod,
                            pcl_next,
                            velocity_remaining,
                        )
                        step_index = it * self.num_modules + mod
                        remaining_fraction = float(
                            total_steps - step_index
                        ) / float(total_steps)
                        raw_gate = self._adaptive_stage_gate(
                            features,
                            mod,
                            remaining_fraction,
                        )
                        gate = 1.0 + self.adaptive_stage_stop_strength * (
                            raw_gate - 1.0
                        )
                    else:
                        pred_dir = self._velocity(
                            mod, pcl_next, velocity_remaining
                        )
                    step = (
                        (1.0 / its)
                        * (1.0 / self.num_modules)
                        * pred_d
                        * pred_dir
                    )
                    if gate is not None:
                        step = gate * step
                    pcl_next = pcl_next + step
        return pcl_next, None

    def _rand_rot(self, seed):
        import numpy as np
        rng = np.random.RandomState(seed)
        q, _ = np.linalg.qr(rng.randn(3, 3))
        if np.linalg.det(q) < 0:
            q[:, 0] = -q[:, 0]
        return jt.array(q.astype('float32'))

    def _denoise_cloud(self, pc_noisy):
        # cloud-level multi-pass: feed the denoised result back through denoising
        pc = pc_noisy
        for _ in range(max(1, self.predict_passes)):
            pc = patch_based_denoise(model=self, pcl_noisy=pc, patch_size=1000, seed_k=6,
                                     seed_k_alpha=1, fusion=self.predict_fusion)
        return pc

    @jt.no_grad()
    def predict_step(self, batch: Dict) -> List[Dict]:
        pc_noisy_batch = batch['pc_noisy']
        if not isinstance(pc_noisy_batch, jt.Var):
            pc_noisy_batch = jt.array(pc_noisy_batch)
        assert pc_noisy_batch.ndim == 3
        res = []
        for pc_noisy in pc_noisy_batch:
            if self.predict_tta > 0:
                # denoise several rotated copies, rotate back, average (point order preserved)
                acc = None
                for r in range(self.predict_tta):
                    R = self._rand_rot(r + 1)
                    den = self._denoise_cloud(jt.matmul(pc_noisy, R))
                    den = jt.matmul(den, R.transpose())
                    acc = den if acc is None else acc + den
                pc_denoised = acc / self.predict_tta
            else:
                pc_denoised = self._denoise_cloud(pc_noisy)
            res.append({"pc_denoised": pc_denoised.detach().numpy()})
        return res

    def process_fn(self, batch: List[Asset]) -> List[Dict]:
        res = []
        for b in batch:
            if not self.is_predict():
                assert b.meta is not None
                item = {
                    "pcl_clean": b.meta['pcl_clean'],
                    "pcl_noisy_L2": b.meta['pcl_noisy_L2'],
                    "seed_points_t": b.meta['seed_points_t'],
                    "original_time_step": b.meta['original_time_step'],
                }
                if self.distance_noise_adaptive:
                    assert 'noise_std' in b.meta
                    item['noise_std'] = np.asarray(
                        b.meta['noise_std'], dtype=np.float32
                    ).reshape(1)
                res.append(item)
            else:
                res.append({"pc_noisy": b.sampled_vertices_noisy})
        return res
