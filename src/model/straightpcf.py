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


def _make_decoder(decoder_type, z_dim, out_dim, hidden_size):
    """HybridPF dynamic graph-conv decoder vs the plain MLP decoder."""
    if decoder_type == 'graph':
        return GraphConvDecoder(z_dim=z_dim, dim=3, out_dim=out_dim, hidden_size=hidden_size)
    return Decoder(z_dim=z_dim, dim=3, out_dim=out_dim, hidden_size=hidden_size)


class VelocityNet(nn.Module):
    """Encoder (EdgeConv) + Decoder predicting a 3D flow direction."""

    def __init__(self, frame_knn, feat_embedding_dim, decoder_hidden_dim, decoder_type='mlp'):
        super().__init__()
        self.encoder = FeatureExtraction(k=frame_knn, input_dim=3, embedding_dim=feat_embedding_dim)
        self.decoder = _make_decoder(decoder_type, self.encoder.embedding_dim, 3, decoder_hidden_dim)

    def execute(self, x):
        B, N, d = x.shape
        feat = self.encoder(x)
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
        edim = cfg['feat_embedding_dim']
        hdim = cfg['decoder_hidden_dim']

        self.decoder_type = cfg.get('decoder_type', 'mlp')  # 'mlp' | 'graph' (HybridPF)
        self.num_modules = 1 if self.stage == 'vm' else cfg.get('num_modules', 2)
        self.velocity_nets = nn.ModuleList(
            [VelocityNet(self.frame_knn, edim, hdim, self.decoder_type) for _ in range(self.num_modules)]
        )

        if self.stage == 'spcf':
            # DistanceModule: predicts a scalar in (0,1) scaling the trajectory.
            self.encoder = FeatureExtraction(
                k=self.frame_knn, input_dim=3, embedding_dim=edim,
                distance_estimation=cfg.get('distance_estimation', True),
            )
            self.decoder = _make_decoder(self.decoder_type, edim, 1, hdim)

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
        loss = (((pred[:, idx, :] - target[:, idx, :]) ** 2.0) / self.dsm_sigma).sum(dim=-1).mean()
        return loss

    def _loss_cvm(self, pcl_clean, pcl_noisy_L2, seeds_t, t):
        B, N, d = pcl_noisy_L2.shape
        grad_target = pcl_clean - pcl_noisy_L2
        total_dir = 0.0
        total_cons = 0.0
        curr = (t * self.num_modules) / self.num_modules
        curr = curr.reshape(B, 1, 1)
        pcl_noisy = curr * pcl_clean + (1 - curr) * pcl_noisy_L2
        pcl_noisy = pcl_noisy - seeds_t
        for mod in range(self.num_modules):
            pred_dir = self.velocity_nets[mod](pcl_noisy)
            total_dir = total_dir + ((pred_dir - grad_target) ** 2).sum(dim=-1).mean()
            pcl_noisy = pcl_noisy + ((1.0 - t.reshape(B, 1, 1)) / self.num_modules) * pred_dir
            if mod < self.num_modules - 1:
                cp1 = (t * (self.num_modules - (mod + 1)) + (mod + 1)) / self.num_modules
                cp1 = cp1.reshape(B, 1, 1)
                interp = cp1 * pcl_clean + (1 - cp1) * pcl_noisy_L2 - seeds_t
                total_cons = total_cons + ((interp - pcl_noisy) ** 2).sum(dim=-1).mean()
        return (total_dir + 10 * total_cons) / self.dsm_sigma

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
        pred_d = self.decoder(c=feat_d.reshape(-1, F_d), B=B, N=N).reshape(B)
        loss = ((pred_d - ratio) ** 2).mean()

        cur = pcl_noisy_c
        for mod in range(self.num_modules):
            pred_dir = self.velocity_nets[mod](cur)
            cur = cur + (1.0 / self.num_modules) * pred_d.reshape(B, 1, 1) * pred_dir
        finetune = 2e2 * ((pcl_clean_c - cur) ** 2).sum(dim=-1).mean()
        return (loss + finetune) / self.dsm_sigma

    # ----- inference (called per-patch by patch_based_denoise) -----
    def denoise_langevin_dynamics(self, pcl_noisy, num_steps: int = None):
        B, N, d = pcl_noisy.shape
        with jt.no_grad():
            self.eval()
            pcl_next = pcl_noisy.clone()
            if self.stage == 'spcf':
                feat_d = self.encoder(pcl_next)
                F_d = feat_d.shape[2]
                pred_d = self.decoder(c=feat_d.reshape(-1, F_d), B=B, N=N).reshape(B, 1, 1)
            else:
                pred_d = jt.ones((B, 1, 1))
            its = self.tot_its
            for _ in range(its):
                for mod in range(self.num_modules):
                    pred_dir = self.velocity_nets[mod](pcl_next)
                    pcl_next = pcl_next + (1.0 / its) * (1.0 / self.num_modules) * pred_d * pred_dir
        return pcl_next, None

    @jt.no_grad()
    def predict_step(self, batch: Dict) -> List[Dict]:
        pc_noisy_batch = batch['pc_noisy']
        if not isinstance(pc_noisy_batch, jt.Var):
            pc_noisy_batch = jt.array(pc_noisy_batch)
        assert pc_noisy_batch.ndim == 3
        res = []
        for pc_noisy in pc_noisy_batch:
            pc_denoised = patch_based_denoise(
                model=self, pcl_noisy=pc_noisy, patch_size=1000, seed_k=6, seed_k_alpha=1,
            )
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
