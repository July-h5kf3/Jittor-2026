"""Jittor port of the frozen 3DMambaIPF feature/displacement network."""

from __future__ import annotations

from typing import Optional, Tuple

import jittor as jt
from jittor import nn

from ..ops.geometry import knn_indices
from .edgeconv import DynamicEdgeConv
from .mamba import MixerModel


class FeatureExtraction(nn.Module):
    def __init__(
        self,
        k: int = 32,
        input_dim: int = 3,
        z_dim: int = 0,
        embedding_dim: int = 512,
        output_dim: int = 3,
    ) -> None:
        super().__init__()
        self.k = int(k)
        self.input_dim = int(input_dim)
        self.z_dim = int(z_dim)
        self.embedding_dim = int(embedding_dim)
        self.output_dim = int(output_dim)

        self.conv1 = DynamicEdgeConv(3, 16)
        self.conv2 = DynamicEdgeConv(16, 48)
        self.conv3 = DynamicEdgeConv(48, 144)
        self.conv4 = DynamicEdgeConv(16 + 48 + 144, self.embedding_dim)

        self.mamba_1 = MixerModel(d_model=16, n_layer=6)
        self.mamba_2 = MixerModel(d_model=48, n_layer=6)
        self.mamba_3 = MixerModel(d_model=144, n_layer=6)
        self.mamba_4 = MixerModel(d_model=512, n_layer=1)
        self.mamba_5 = MixerModel(d_model=256, n_layer=1)
        self.mamba_6 = MixerModel(d_model=128, n_layer=1)

        self.linear1 = nn.Linear(self.embedding_dim, 256, bias=False)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, self.output_dim)

        if self.z_dim > 0:
            self.linear_proj = nn.Linear(512, self.z_dim)
            self.dropout_proj = nn.Dropout(0.1)

    def _neighbours(self, features: jt.Var) -> jt.Var:
        if features.shape[1] <= self.k:
            raise ValueError(
                f"point count {features.shape[1]} must exceed frame_knn {self.k}"
            )
        _, indices = knn_indices(features.float32(), features.float32(), self.k + 1)
        # Coordinates/features are continuous in the competition data, so the first
        # zero-distance entry is the query itself. Dropping it is equivalent to the
        # original k+1 KNN followed by remove_self_loops.
        return indices[:, :, 1:]

    def execute(
        self,
        points: jt.Var,
        displacement_features: Optional[jt.Var] = None,
    ) -> Tuple[jt.Var, Optional[jt.Var]]:
        if points.ndim != 3 or points.shape[2] != 3:
            raise ValueError("FeatureExtraction expects (B,N,3)")
        if displacement_features is not None:
            if self.z_dim <= 0:
                raise ValueError("this frozen model has z_dim=0")
            projected = nn.relu(self.linear_proj(displacement_features))
            projected = self.dropout_proj(projected)
            points = jt.concat([points, projected], dim=-1)

        neighbours = self._neighbours(points)
        x1 = self.conv1(points, neighbours)
        x1 = self.mamba_1(x1)

        neighbours = self._neighbours(x1)
        x2 = self.conv2(x1, neighbours)
        x2 = self.mamba_2(x2)

        neighbours = self._neighbours(x2)
        x3 = self.conv3(x2, neighbours)
        x3 = self.mamba_3(x3)

        neighbours = self._neighbours(x3)
        combined = jt.concat([x1, x2, x3], dim=-1)
        output = self.conv4(combined, neighbours)
        output = self.mamba_4(output)
        output = nn.relu(self.mamba_5(self.linear1(output)))
        output = nn.relu(self.mamba_6(self.linear2(output)))
        output = jt.tanh(self.linear3(output))

        if self.z_dim > 0:
            return output, combined.permute(0, 2, 1).contiguous()
        return output, None
