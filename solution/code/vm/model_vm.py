"""Jittor StraightPCF velocity module used by B-board VM inference and training."""

from __future__ import annotations

import numpy as np
import jittor as jt
from jittor import nn

FRAME_KNN = 32
EMBEDDING_DIM = 256
DECODER_HIDDEN_DIM = 64
DSM_SIGMA = 0.01
TRAIN_POINTS = 128


def knn_indices(x, k: int):
    batch, count, _ = x.shape
    distance = ((x.unsqueeze(2) - x.unsqueeze(1)) ** 2).sum(-1)
    diag = jt.array(np.eye(count, dtype=np.float32) * 1e30).unsqueeze(0)
    distance = distance + diag
    _, indices = jt.topk(distance, k, dim=-1, largest=False)
    return indices


class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation="ReLU"):
        super().__init__()
        if activation == "ReLU":
            self.mlp = nn.Sequential(
                nn.Linear(2 * in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
                nn.ReLU(),
            )
            self.lin = nn.Sequential(nn.Linear(in_channels, out_channels), nn.ReLU())
        elif activation is None:
            self.mlp = nn.Sequential(
                nn.Linear(2 * in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
            )
            self.lin = nn.Linear(in_channels, out_channels)
        else:
            raise ValueError(activation)

    def execute(self, x, neighbor_ids):
        batch, count, channels = x.shape
        k = int(neighbor_ids.shape[-1])
        flat = neighbor_ids.reshape(batch, count * k)
        gathered = x.reindex(
            [batch, count * k, channels],
            ["i0", "@e0(i0,i1)", "i2"],
            extras=[flat],
        ).reshape(batch, count, k, channels)
        centers = x.unsqueeze(2).broadcast(gathered.shape)
        messages = self.mlp(jt.concat([centers, gathered - centers], dim=-1))
        pooled = messages.max(dim=2)
        return pooled + self.lin(x)


class FeatureExtraction(nn.Module):
    def __init__(self):
        super().__init__()
        self.k = FRAME_KNN
        self.conv1 = EdgeConv(3, 32)
        self.conv2 = EdgeConv(32, 64)
        self.conv3 = EdgeConv(96, EMBEDDING_DIM, activation=None)

    def execute(self, x):
        x1 = self.conv1(x, knn_indices(x, self.k))
        x2 = self.conv2(x1, knn_indices(x1, self.k))
        combined = jt.concat([x1, x2], dim=-1)
        return self.conv3(combined, knn_indices(x2, self.k))


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin_1 = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        self.bn_1_out = nn.BatchNorm1d(EMBEDDING_DIM)
        self.lin_2 = nn.Linear(EMBEDDING_DIM, DECODER_HIDDEN_DIM)
        self.bn_2_out = nn.BatchNorm1d(DECODER_HIDDEN_DIM)
        self.lin_3 = nn.Linear(DECODER_HIDDEN_DIM, 3)
        self.actvn_out = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def execute(self, c):
        net = self.dropout(self.actvn_out(self.bn_1_out(self.lin_1(c))))
        net = self.dropout(self.actvn_out(self.bn_2_out(self.lin_2(net))))
        return self.lin_3(net)


class VelocityModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FeatureExtraction()
        self.decoder = Decoder()

    def execute(self, interpolated, noisy_l2, clean, point_ids):
        features = self.encoder(interpolated)
        selected = features[:, point_ids, :]
        prediction = self.decoder(selected.reshape(-1, EMBEDDING_DIM)).reshape(
            interpolated.shape[0], point_ids.shape[0], 3
        )
        target = clean[:, point_ids, :] - noisy_l2[:, point_ids, :]
        return (((prediction - target) ** 2) / DSM_SIGMA).sum(dim=-1).mean()

    def denoise_langevin(self, pcl_noisy, num_steps: int = 4):
        pcl_next = pcl_noisy
        for _ in range(num_steps):
            feat = self.encoder(pcl_next)
            pred = self.decoder(feat.reshape(-1, EMBEDDING_DIM)).reshape(
                pcl_next.shape[0], pcl_next.shape[1], 3
            )
            pcl_next = pcl_next + pred / float(num_steps)
        return pcl_next
