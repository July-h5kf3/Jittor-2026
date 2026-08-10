"""Dynamic EdgeConv with the parameter layout of the frozen PyTorch model."""

from __future__ import annotations

import jittor as jt
from jittor import nn

from ..ops.geometry import gather_points


class EdgeConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.mlp = nn.Sequential(
            nn.Linear(2 * self.in_channels, self.out_channels),
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU(),
            nn.Linear(self.out_channels, self.out_channels),
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU(),
        )
        self.lin = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels),
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU(),
        )

    def execute(self, features: jt.Var, neighbour_indices: jt.Var) -> jt.Var:
        """Aggregate edge messages by maximum.

        Args:
            features: ``(B,N,C)`` node features.
            neighbour_indices: ``(B,N,K)`` neighbours for each destination node.
        """
        if features.ndim != 3 or neighbour_indices.ndim != 3:
            raise ValueError("EdgeConv expects features=(B,N,C), indices=(B,N,K)")
        batch, count, channels = features.shape
        if channels != self.in_channels or neighbour_indices.shape[:2] != (batch, count):
            raise ValueError("EdgeConv feature/index shape mismatch")
        neighbours = gather_points(features, neighbour_indices)
        k = neighbour_indices.shape[2]
        centers = features.unsqueeze(2).broadcast((batch, count, k, channels))
        messages = jt.concat([centers, neighbours - centers], dim=-1)
        messages = self.mlp(messages.reshape(batch * count * k, 2 * channels))
        messages = messages.reshape(batch, count, k, self.out_channels)
        aggregated = jt.max(messages, dim=2)
        residual = self.lin(features.reshape(batch * count, channels))
        residual = residual.reshape(batch, count, self.out_channels)
        return aggregated + residual


class DynamicEdgeConv(EdgeConv):
    pass
