from typing import Optional
from jittor import nn

import jittor as jt
import numpy as np

class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation: Optional[str]='ReLU'):
        super().__init__()
        
        if activation == 'ReLU':
            self.mlp = nn.Sequential(
                nn.Linear(2 * in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
                nn.ReLU()
            )
            self.lin = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU()
            )
        elif activation is None:
            self.mlp = nn.Sequential(
                nn.Linear(2 * in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
            )
            self.lin = nn.Linear(in_channels, out_channels)
        else:
            raise Exception("Please assign valid activation to MLP!")
    
    def execute(self, x, edge_index):
        """
        x: (N, C)
        edge_index: (2, E)
        """
        src = edge_index[0]  # (E,)
        dst = edge_index[1]  # (E,)
        
        # gather
        x_i = x[dst]  # (E, C)
        x_j = x[src]  # (E, C)
        
        # message
        tmp = jt.concat([x_i, x_j - x_i], dim=1)  # (E, 2C)
        msg = self.mlp(tmp)  # (E, out_channels)
        
        N = x.shape[0]
        out = jt.full((N, msg.shape[1]), 0)
        cnt = jt.full((N, msg.shape[1]), 0)
        
        # scatter mean
        out = out.scatter_(0, dst.unsqueeze(1).broadcast(msg.shape), msg, reduce='add')
        cnt = cnt.scatter_(0, dst.unsqueeze(1).broadcast(msg.shape), jt.ones_like(msg), reduce='add')
        out = out / (cnt + 1)
        out_2 = self.lin(x)
        return out + out_2

class DynamicEdgeConv(EdgeConv):
    def __init__(self, in_channels, out_channels, activation: Optional[str]='ReLU'):
        super().__init__(in_channels, out_channels, activation)
    
    def execute(self, x, edge_index):
        return super().execute(x, edge_index)

class SelfAttentionBlock(nn.Module):
    """Multi-head self-attention over all patch points (global context) + FFN,
    each with a residual connection and LayerNorm. EdgeConv only aggregates local
    kNN; this lets every point attend to the whole patch."""

    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.dh = dim // heads
        self.scale = self.dh ** -0.5
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def execute(self, x):
        # x: (B, N, D)
        B, N, D = x.shape
        h, dh = self.heads, self.dh
        q = self.q(x).reshape(B, N, h, dh).permute(0, 2, 1, 3)  # (B,h,N,dh)
        k = self.k(x).reshape(B, N, h, dh).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, h, dh).permute(0, 2, 1, 3)
        attn = jt.matmul(q, k.permute(0, 1, 3, 2)) * self.scale   # (B,h,N,N)
        attn = nn.softmax(attn, dim=-1)
        out = jt.matmul(attn, v).permute(0, 2, 1, 3).reshape(B, N, D)
        x = self.norm1(x + self.proj(out))
        x = self.norm2(x + self.ff(x))
        return x


def _gather_neighbours(feat, idx):
    """Gather source features for either self-kNN or cross-set queries."""
    B, source_count, C = feat.shape
    query_count = idx.shape[1]
    k = idx.shape[2]
    base = (jt.arange(B) * source_count).reshape(B, 1, 1)
    flat = (idx + base).reshape(-1)
    return feat.reshape(B * source_count, C)[flat].reshape(B, query_count, k, C)


class PointTransformerBlock(nn.Module):
    """Point Transformer (Zhao et al. 2021) vector self-attention: local kNN in
    POSITION space, relative-position encoding delta=MLP(p_i-p_j), and subtraction
    based vector attention (per-channel weights). Position-aware, unlike plain
    dot-product attention. Residual + LayerNorm."""

    def __init__(self, dim, k=16):
        super().__init__()
        self.k = k
        self.lin_q = nn.Linear(dim, dim)
        self.lin_k = nn.Linear(dim, dim)
        self.lin_v = nn.Linear(dim, dim)
        # relative-position encoding input = [dx,dy,dz, ||d||] (radial term is
        # rotation-invariant and a strong cue for local surface geometry).
        self.pos_mlp = nn.Sequential(nn.Linear(4, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.attn_mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)

    def execute(self, x, pos):
        # x: (B,N,C), pos: (B,N,3)
        B, N, C = x.shape
        idx = get_knn_idx(pos, pos, self.k)             # (B,N,k) local nbrs by position
        q = self.lin_q(x)                                # (B,N,C)
        k_nb = _gather_neighbours(self.lin_k(x), idx)    # (B,N,k,C)
        v_nb = _gather_neighbours(self.lin_v(x), idx)    # (B,N,k,C)
        pos_nb = _gather_neighbours(pos, idx)            # (B,N,k,3)
        rel = pos.unsqueeze(2) - pos_nb                  # (B,N,k,3)
        dist = jt.sqrt((rel ** 2).sum(-1, keepdims=True) + 1e-8)  # (B,N,k,1)
        delta = self.pos_mlp(jt.concat([rel, dist], dim=-1))      # (B,N,k,C)
        attn = self.attn_mlp(q.unsqueeze(2) - k_nb + delta)  # (B,N,k,C) vector attention
        attn = nn.softmax(attn, dim=2)                   # over k neighbours
        out = (attn * (v_nb + delta)).sum(dim=2)         # (B,N,C)
        return self.norm(x + out)


class NoiseAwareRoPEAttention(nn.Module):
    """Noise-Aware 3D RoPE local attention (revised per infer.md critique).

    Fixes over the naive version:
    1. Content/position channel split: only POSITION channels carry RoPE+gating;
       CONTENT channels go through a plain dot product. So a noisy point's content
       info is never zeroed out by the noise gate (infer.md S6).
    2. 3D-GS-RoPE: directions are NOT free-learned (which the PointTransformerX
       ablation shows is the weakest variant). Instead one learnable ORTHONORMAL
       basis B (in SO(3)) rotates the coordinate, then axial RoPE per axis x
       multi-scale frequencies omega_k -> xi = omega_k * b_a (infer.md S3).
    3. Noise-adaptive gating g_{i,r}=exp(-0.5 sigma_i^2 ||xi_r||^2) on position
       channels only, sigma from a learned per-point noise head.
    Local kNN dot-product attention, residual + LayerNorm. Translation invariant.
    """

    def __init__(self, dim, k=16):
        super().__init__()
        self.k = k
        self.dim = dim
        self.n_scales = max(1, (dim // 2) // 6)        # frequencies per axis
        self.n_pairs = 3 * self.n_scales                # rotated channel pairs (3 axes)
        self.n_pos = 2 * self.n_pairs                   # position channels
        self.n_content = dim - self.n_pos               # plain content channels
        self.scale = dim ** -0.5
        self.lin_q = nn.Linear(dim, dim)
        self.lin_k = nn.Linear(dim, dim)
        self.lin_v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.noise_head = nn.Sequential(nn.Linear(dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, 1))
        # learnable orthonormal coordinate frame (orthonormalised via QR in forward)
        self.basis_raw = jt.array((np.eye(3) + 0.05 * np.random.randn(3, 3)).astype('float32'))
        # multi-scale frequencies (log-spaced), learnable
        self.omega = jt.array(np.exp(np.linspace(0.0, 3.0, self.n_scales)).astype('float32'))

    def _orthonormal(self, M):
        # differentiable Gram-Schmidt on a 3x3 matrix (rows) -> orthonormal frame
        # (avoids jt.linalg.qr, which needs cupy)
        def n(v):
            return v / (jt.sqrt((v ** 2).sum()) + 1e-8)
        v0 = n(M[0])
        v1 = n(M[1] - (M[1] * v0).sum() * v0)
        v2 = n(M[2] - (M[2] * v0).sum() * v0 - (M[2] * v1).sum() * v1)
        return jt.stack([v0, v1, v2], dim=0)

    def _rope_gate(self, t, cos, sin, g):
        # t: (B,N,n_pos); cos/sin/g: (B,N,n_pairs). Rotate each pair by its phase, gate.
        B, N, _ = t.shape
        t2 = t.reshape(B, N, self.n_pairs, 2)
        t0, t1 = t2[..., 0], t2[..., 1]
        r0 = t0 * cos - t1 * sin
        r1 = t0 * sin + t1 * cos
        rot = jt.stack([r0, r1], dim=-1) * g.unsqueeze(-1)
        return rot.reshape(B, N, self.n_pos)

    def execute(self, x, pos):
        B, N, C = x.shape
        # center + patch-shared scale -> translation invariant & scale stable
        center = pos.mean(dim=1, keepdims=True)
        p = pos - center
        s = jt.sqrt((p ** 2).sum(-1).mean(dim=1, keepdims=True)) + 1e-6      # (B,1)
        u = p / s.unsqueeze(-1)                                              # (B,N,3)
        Bm = self._orthonormal(self.basis_raw)                               # (3,3) orthonormal
        uh = jt.matmul(u, Bm.transpose())                                    # coords in learned frame
        # axial phases: pair (axis a, scale k) -> omega_k * uh[...,a]
        theta = jt.concat([uh[..., a:a + 1] * self.omega.reshape(1, 1, -1) for a in range(3)], dim=-1)
        cos, sin = jt.cos(theta), jt.sin(theta)                             # (B,N,n_pairs)
        sigma = nn.softplus(self.noise_head(x))                            # (B,N,1)
        freq_n2 = jt.concat([self.omega ** 2, self.omega ** 2, self.omega ** 2]).reshape(1, 1, self.n_pairs)
        g = jt.exp(-0.5 * (sigma ** 2) * freq_n2)                          # (B,N,n_pairs)
        q, k, v = self.lin_q(x), self.lin_k(x), self.lin_v(x)
        # split content (plain) | position (rope+gate)
        qc, qp = q[..., :self.n_content], q[..., self.n_content:]
        kc, kp = k[..., :self.n_content], k[..., self.n_content:]
        q2 = jt.concat([qc, self._rope_gate(qp, cos, sin, g)], dim=-1)
        k2 = jt.concat([kc, self._rope_gate(kp, cos, sin, g)], dim=-1)
        idx = get_knn_idx(pos, pos, self.k)
        k_nb = _gather_neighbours(k2, idx)                                  # (B,N,k,C)
        v_nb = _gather_neighbours(v, idx)
        score = (q2.unsqueeze(2) * k_nb).sum(-1) * self.scale               # (B,N,k)
        attn = nn.softmax(score, dim=2)
        out = (attn.unsqueeze(-1) * v_nb).sum(2)
        return self.norm(x + self.proj(out))


class PointNeXtLiteHierarchy(nn.Module):
    """A narrow hierarchical residual branch for fixed-size point patches.

    The input ordering produced by patch extraction is distance-sorted, so a
    strided subset gives a cheap, deterministic coarse set without a Python FPS
    loop. Coarse features pool position-kNN fine features, receive one local
    residual update, and are interpolated back with inverse-distance 3-NN.
    The final projection is zero-initialized so adding this branch preserves a
    loaded EdgeConv checkpoint exactly at initialization.
    """

    def __init__(self, dim, hidden_dim=64, stride=4, k=16, interpolation_k=3):
        super().__init__()
        if stride < 2:
            raise ValueError("PointNeXt-lite stride must be at least 2")
        if k < 1 or interpolation_k < 1:
            raise ValueError("PointNeXt-lite neighbour counts must be positive")
        self.hidden_dim = hidden_dim
        self.stride = stride
        self.k = k
        self.interpolation_k = interpolation_k
        self.reduce = nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU())
        message_dim = hidden_dim + 4
        self.pool_mlp = nn.Sequential(
            nn.Linear(message_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pool_skip = nn.Linear(hidden_dim, hidden_dim)
        self.coarse_mlp = nn.Sequential(
            nn.Linear(message_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.coarse_out = nn.Linear(hidden_dim, hidden_dim)
        self.fuse = nn.Linear(2 * hidden_dim, dim)
        self.fuse.weight.assign(jt.zeros_like(self.fuse.weight))
        self.fuse.bias.assign(jt.zeros_like(self.fuse.bias))

    @staticmethod
    def _relative_geometry(center_pos, neighbour_pos):
        rel = neighbour_pos - center_pos.unsqueeze(2)
        dist = jt.sqrt((rel ** 2).sum(-1, keepdims=True) + 1e-12)
        radius = jt.maximum(dist.max(dim=2, keepdims=True), jt.ones_like(dist[:, :, :1]) * 1e-6)
        return jt.concat([rel / radius, dist / radius], dim=-1)

    def _aggregate(self, center_pos, center_feat, source_pos, source_feat, k, offset, mlp):
        available = source_pos.shape[1] - offset
        k = min(k, available)
        idx = get_knn_idx(center_pos, source_pos, k, offset=offset)
        neighbour_pos = _gather_neighbours(source_pos, idx)
        neighbour_feat = _gather_neighbours(source_feat, idx)
        geometry = self._relative_geometry(center_pos, neighbour_pos)
        message = jt.concat(
            [neighbour_feat - center_feat.unsqueeze(2), geometry],
            dim=-1,
        )
        return jt.max(mlp(message), dim=2)

    def execute(self, x, pos):
        # x: (B,N,C), pos: (B,N,3)
        fine = self.reduce(x)
        coarse_pos = pos[:, ::self.stride, :]
        coarse_seed = fine[:, ::self.stride, :]
        coarse = self.pool_skip(coarse_seed) + self._aggregate(
            coarse_pos,
            coarse_seed,
            pos,
            fine,
            self.k,
            0,
            self.pool_mlp,
        )
        if coarse_pos.shape[1] > 1:
            coarse_message = self._aggregate(
                coarse_pos,
                coarse,
                coarse_pos,
                coarse,
                self.k,
                1,
                self.coarse_mlp,
            )
            coarse = coarse + self.coarse_out(nn.relu(coarse_message))

        interp_k = min(self.interpolation_k, coarse_pos.shape[1])
        interp_idx = get_knn_idx(pos, coarse_pos, interp_k)
        neighbour_pos = _gather_neighbours(coarse_pos, interp_idx)
        neighbour_feat = _gather_neighbours(coarse, interp_idx)
        dist = jt.sqrt(((pos.unsqueeze(2) - neighbour_pos) ** 2).sum(-1) + 1e-12)
        weight = 1.0 / (dist + 1e-6)
        weight = weight / (weight.sum(dim=2, keepdims=True) + 1e-8)
        interpolated = (neighbour_feat * weight.unsqueeze(-1)).sum(dim=2)
        return x + self.fuse(jt.concat([fine, interpolated], dim=-1))


class FeatureExtraction(nn.Module):
    def __init__(
        self,
        k=32,
        input_dim=0,
        embedding_dim=512,
        distance_estimation=False,
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

        self.k = k
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.encoder_type = encoder_type
        if encoder_type not in ('edgeconv', 'pointnext_lite'):
            raise ValueError(f"unsupported encoder_type: {encoder_type}")
        self.distance_estimation = distance_estimation
        self.multiscale = multiscale
        self.ms_ks = [16, 32]   # C3 multi-scale encoder kNN scales (fine+base; max=baseline -> no extra peak mem)
        self.film = film        # C4: FiLM conditioning on a noise/roughness proxy
        self.condition_dim = int(condition_dim or 0)
        if film:
            self.film_fc1 = nn.Linear(1, max(8, embedding_dim // 2))
            self.film_fc2 = nn.Linear(max(8, embedding_dim // 2), 2 * embedding_dim)
            self.film_fc2.weight.assign(jt.zeros_like(self.film_fc2.weight))  # gamma=beta=0 at start
            self.film_fc2.bias.assign(jt.zeros_like(self.film_fc2.bias))
        if self.condition_dim > 0:
            self.condition_fc1 = nn.Linear(self.condition_dim, max(8, embedding_dim // 2))
            self.condition_fc2 = nn.Linear(max(8, embedding_dim // 2), 2 * embedding_dim)
            self.condition_fc2.weight.assign(jt.zeros_like(self.condition_fc2.weight))
            self.condition_fc2.bias.assign(jt.zeros_like(self.condition_fc2.bias))

        self.conv1 = DynamicEdgeConv(self.input_dim, embedding_dim // 8)
        self.conv2 = DynamicEdgeConv(embedding_dim // 8, embedding_dim // 4)
        self.conv3 = DynamicEdgeConv(
            embedding_dim // 8 + embedding_dim // 4,
            embedding_dim,
            activation=None
        )
        self.attention = attention
        if attention:
            self.attn = NoiseAwareRoPEAttention(embedding_dim, k=16)
        if encoder_type == 'pointnext_lite':
            self.hierarchy = PointNeXtLiteHierarchy(
                dim=embedding_dim,
                hidden_dim=hierarchy_hidden_dim,
                stride=hierarchy_stride,
                k=hierarchy_k,
            )

    # ========= edge_index 构建 =========
    def get_edge_index(self, x, k=None):
        # x: (B, N, C)
        k = self.k if k is None else k
        B, N, _ = x.shape
        knn_idx = get_knn_idx(x, x, k + 1)  # (B, N, k+1)
        knn_idx = knn_idx[:, :, 1:]
        base = jt.arange(B) * N  # (B,)
        base = base.reshape(B, 1, 1)
        
        knn_idx = knn_idx + base  # (B, N, k)
        
        dst = jt.arange(N)
        dst = dst.reshape(1, N, 1).broadcast((B, N, k))
        dst = dst + base
        
        src = knn_idx.reshape(-1)
        dst = dst.reshape(-1)
        
        edge_index = jt.stack([src, dst], dim=0)  # (2, E)
        
        return edge_index
    
    def normalize_patch(self, pcl):
        scale = jt.sqrt((pcl ** 2).sum(-1, keepdims=True))
        scale = scale.max(dim=-2, keepdims=True)
        return pcl / (scale + 1e-8) # type: ignore

    def _sigma_hat(self, x):
        # scale-invariant roughness proxy (B,1): mean 4-NN distance / patch radius
        B, N, _ = x.shape
        idx = get_knn_idx(x, x, 4, offset=1)                 # (B,N,4) nearest non-self
        nb = _gather_neighbours(x, idx)                       # (B,N,4,3)
        d = jt.sqrt(((x.unsqueeze(2) - nb) ** 2).sum(-1) + 1e-12).mean(dim=2).mean(dim=1, keepdims=True)
        c = x.mean(dim=1, keepdims=True)                      # (B,1,3) centroid
        r = jt.sqrt(((x - c) ** 2).sum(-1).mean(dim=1, keepdims=True) + 1e-12)  # (B,1) patch radius
        return d / (r + 1e-8)                                 # (B,1) dimensionless
    
    def execute(self, x, condition=None):
        # x: (B, N, C)
        B, N, _ = x.shape
        
        if self.distance_estimation:
            x = self.normalize_patch(x)

        pos = x  # (B,N,3) positions for position-aware attention

        # -------- conv (serial, or C3 multi-scale average) --------
        x_flat = x.reshape(B * N, -1)
        if self.multiscale:
            ms = self.ms_ks
            x1 = sum(self.conv1(x_flat, self.get_edge_index(x, k=kk)).reshape(B, N, -1) for kk in ms) / len(ms)
            x1_flat = x1.reshape(B * N, -1)
            x2 = sum(self.conv2(x1_flat, self.get_edge_index(x1, k=kk)).reshape(B, N, -1) for kk in ms) / len(ms)
            x_combined = jt.concat([x1, x2], dim=-1)
            x_combined_flat = x_combined.reshape(B * N, -1)
            x3 = sum(self.conv3(x_combined_flat, self.get_edge_index(x2, k=kk)).reshape(B, N, -1) for kk in ms) / len(ms)
        else:
            edge_index = self.get_edge_index(x)
            x1 = self.conv1(x_flat, edge_index)
            x1 = x1.reshape(B, N, -1)
            edge_index = self.get_edge_index(x1)
            x1_flat = x1.reshape(B * N, -1)
            x2 = self.conv2(x1_flat, edge_index)
            x2 = x2.reshape(B, N, -1)
            edge_index = self.get_edge_index(x2)
            x_combined = jt.concat([x1, x2], dim=-1)
            x_combined_flat = x_combined.reshape(B * N, -1)
            x3 = self.conv3(x_combined_flat, edge_index)
            x3 = x3.reshape(B, N, -1)

        if self.attention:
            x3 = self.attn(x3, pos)   # position-aware (Point Transformer) attention

        if self.encoder_type == 'pointnext_lite':
            x3 = self.hierarchy(x3, pos)

        if self.film:
            sig = self._sigma_hat(pos)                        # (B,1)
            gb = self.film_fc2(nn.relu(self.film_fc1(sig)))   # (B, 2F)
            F = self.embedding_dim
            gamma = gb[:, :F].unsqueeze(1)                    # (B,1,F)
            beta = gb[:, F:].unsqueeze(1)
            x3 = (1.0 + gamma) * x3 + beta

        if self.condition_dim > 0:
            if condition is None:
                condition = jt.zeros((B, self.condition_dim))
            condition = condition.reshape(B, self.condition_dim)
            cb = self.condition_fc2(nn.relu(self.condition_fc1(condition)))
            F = self.embedding_dim
            gamma = cb[:, :F].unsqueeze(1)
            beta = cb[:, F:].unsqueeze(1)
            x3 = (1.0 + gamma) * x3 + beta

        return x3

class Decoder(nn.Module):
    
    def __init__(self, z_dim, dim, out_dim, hidden_size, scalar_reduce=True):
        super().__init__()
        self.z_dim = z_dim
        self.dim = dim
        self.out_dim = out_dim
        self.hidden_size = hidden_size
        self.scalar_reduce = scalar_reduce
        c_dim = z_dim
        self.lin_1 = nn.Linear(c_dim, c_dim)
        self.bn_1_out = nn.BatchNorm1d(c_dim)
        
        self.lin_2 = nn.Linear(c_dim, hidden_size)
        self.bn_2_out = nn.BatchNorm1d(hidden_size)
        
        self.lin_3 = nn.Linear(hidden_size, out_dim)
        
        self.actvn_out = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def execute(self, c, B=None, N=None):
        """
        c: (B*N, F)
        """
        net = self.lin_1(c)
        net = self.bn_1_out(net)
        net = self.actvn_out(net)
        net = self.dropout(net)
        
        net = self.lin_2(net)
        net = self.bn_2_out(net)
        net = self.actvn_out(net)
        net = self.dropout(net)
        
        if self.out_dim == 1:
            net = net.reshape(B, N, -1)
            if self.scalar_reduce:
                net = jt.max(net, dim=1, keepdims=True)
            net = self.lin_3(net)
            net = jt.sigmoid(net)
        else:
            net = self.lin_3(net)
        return net

def get_knn_idx(x, y, k, offset=0):
    """
    x: (B, N, d)
    y: (B, M, d)
    return: (B, N, k)
    """
    K = k + offset
    if x.shape[-1] == 3:
        _, idx = jt.misc.knn(x, y, K)
    else:
        dist = ((x.unsqueeze(2) - y.unsqueeze(1)) ** 2).sum(-1)
        _, idx = jt.topk(dist, k=K, dim=-1, largest=False)
    return idx[:, :, offset:]


def _feature_edge_index(z, k):
    """Build a directed kNN edge_index in FEATURE space for a batched (B,N,C) tensor.
    Returns (2, B*N*k) with flattened node ids. dst=center, src=neighbour."""
    B, N, _ = z.shape
    knn_idx = get_knn_idx(z, z, k + 1)[:, :, 1:]          # (B,N,k), drop self
    base = (jt.arange(B) * N).reshape(B, 1, 1)
    src = (knn_idx + base).reshape(-1)
    dst = (jt.arange(N).reshape(1, N, 1).broadcast((B, N, k)) + base).reshape(-1)
    return jt.stack([src, dst], dim=0)


class GraphConvDecoder(nn.Module):
    """HybridPF-style dynamic graph-convolutional decoder.

    Replaces the per-point MLP decoder with EdgeConv layers that aggregate
    topological information in latent feature space (dynamic kNN graph rebuilt
    per layer). Same I/O contract as Decoder: returns (B*N, out_dim) for out_dim>1
    and (B,1,1) for out_dim==1 (global scalar via max-pool + sigmoid).
    """

    def __init__(self, z_dim, dim, out_dim, hidden_size, k=8, scalar_reduce=True):
        super().__init__()
        self.out_dim = out_dim
        self.k = k
        self.scalar_reduce = scalar_reduce
        self.conv1 = EdgeConv(z_dim, hidden_size)
        self.conv2 = EdgeConv(hidden_size, hidden_size, activation=None)
        self.lin_out = nn.Linear(hidden_size, out_dim)
        self.dropout = nn.Dropout(0.1)

    def execute(self, c, B=None, N=None):
        z = c.reshape(B, N, -1)
        ei = _feature_edge_index(z, self.k)
        x = self.conv1(z.reshape(B * N, -1), ei)
        x = self.dropout(x)
        ei2 = _feature_edge_index(x.reshape(B, N, -1), self.k)  # dynamic: rebuild
        x = self.conv2(x, ei2)
        x = self.dropout(x)
        if self.out_dim == 1:
            x = x.reshape(B, N, -1)
            if self.scalar_reduce:
                x = jt.max(x, dim=1, keepdims=True)    # (B,1,H)
            return jt.sigmoid(self.lin_out(x))         # (B,1,1) or (B,N,1)
        return self.lin_out(x)                          # (B*N, out_dim)
