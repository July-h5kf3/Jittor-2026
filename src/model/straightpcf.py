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

    # ----- checkpoint chaining between stages -----
    def init_from_stage(self, ckpt_path: str):
        """Load a previous-stage checkpoint into this stage's submodules.
        vm->cvm: copy the single VM weights into every coupled module.
        cvm->spcf: copy the coupled velocity_nets verbatim (distance module stays fresh).
        """
        prev = jt.load(ckpt_path)
        sd = prev['state_dict'] if isinstance(prev, dict) and 'state_dict' in prev else prev
        # collect source velocity-net sub-state-dicts: keys like 'velocity_nets.<i>.xxx'
        src_mods = {}
        for k, v in sd.items():
            if k.startswith('velocity_nets.'):
                idx = k.split('.')[1]
                src_mods.setdefault(idx, {})[k.split('.', 2)[2]] = v
        if not src_mods:
            raise ValueError(f"no velocity_nets.* found in {ckpt_path}")
        src_keys = sorted(src_mods.keys(), key=int)
        for i in range(self.num_modules):
            # vm(1 module)->cvm(N): broadcast module 0 to all; cvm->spcf: 1-1 map
            src = src_mods[src_keys[i] if i < len(src_keys) else src_keys[0]]
            self.velocity_nets[i].load_state_dict(src)
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
        if self.stage == 'vm':
            loss = self._loss_vm(pcl_clean, pcl_noisy_L2, seeds_t, t)
        elif self.stage == 'cvm':
            loss = self._loss_cvm(pcl_clean, pcl_noisy_L2, seeds_t, t)
        else:
            loss = self._loss_spcf(pcl_clean, pcl_noisy_L2, seeds_t, t)
        return {"loss": loss}

    def execute(self, **kwargs) -> Dict:
        return self.training_step(**kwargs)

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
        return loss

    def _loss_spcf(self, pcl_clean, pcl_noisy_L2, seeds_t, t):
        # velocity nets frozen-ish (eval mode); train the distance module
        self.velocity_nets.eval()
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
            pred_d = pred_d.reshape(B, 1, 1)

        cur = pcl_noisy_c
        velocity_remaining = pred_d.detach()
        for mod in range(self.num_modules):
            pred_dir = self._velocity(mod, cur, velocity_remaining)
            cur = cur + (1.0 / self.num_modules) * pred_d * pred_dir
        finetune = 2e2 * ((pcl_clean_c - cur) ** 2).sum(dim=-1).mean()
        spcf_loss = (loss + finetune) / self.dsm_sigma
        if self.lam_dir > 0 or self.lam_mag > 0 or self.lam_edge > 0:
            # cur is the final distance-scaled endpoint (~ clean_c)
            spcf_loss = spcf_loss + self._aux_losses(pred_dir, pcl_clean_c - pcl_noisy_c, cur, pcl_clean_c)
        return spcf_loss

    # ----- inference (called per-patch by patch_based_denoise) -----
    def denoise_langevin_dynamics(self, pcl_noisy, num_steps: int = None):
        B, N, d = pcl_noisy.shape
        with jt.no_grad():
            self.eval()
            pcl_next = pcl_noisy.clone()
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
            else:
                pred_d = jt.ones((B, 1, 1))
            pred_d = pred_d * self.predict_alpha   # inference distance scale
            its = self.tot_its
            for it in range(its):
                velocity_remaining = pred_d * (float(its - it) / float(its))
                for mod in range(self.num_modules):
                    pred_dir = self._velocity(mod, pcl_next, velocity_remaining)
                    pcl_next = pcl_next + (1.0 / its) * (1.0 / self.num_modules) * pred_d * pred_dir
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
                res.append({
                    "pcl_clean": b.meta['pcl_clean'],
                    "pcl_noisy_L2": b.meta['pcl_noisy_L2'],
                    "seed_points_t": b.meta['seed_points_t'],
                    "original_time_step": b.meta['original_time_step'],
                })
            else:
                res.append({"pc_noisy": b.sampled_vertices_noisy})
        return res
