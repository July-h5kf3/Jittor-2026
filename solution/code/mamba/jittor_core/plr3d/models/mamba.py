"""Pure-Jittor Mamba 1.x block used by the submitted 3DMambaIPF model."""

from __future__ import annotations

import math
from typing import Optional

import jittor as jt
import numpy as np
from jittor import nn

from ..ops.selective_scan import selective_scan


def silu(value: jt.Var) -> jt.Var:
    return value * jt.sigmoid(value)


class Mamba(nn.Module):
    """Mamba 1.1-compatible mixer for the frozen PLR architecture.

    Defaults match ``mamba_ssm.modules.mamba_simple.Mamba`` 1.1.3.post1:
    ``d_state=16``, ``d_conv=4``, ``expand=2`` and automatic ``dt_rank``.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Optional[int] = None,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        layer_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.expand = int(expand)
        self.d_inner = self.expand * self.d_model
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank is None else dt_rank)
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
            bias=True,
        )
        self.x_proj = nn.Linear(
            self.d_inner,
            self.dt_rank + self.d_state * 2,
            bias=False,
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

        dt_std = self.dt_rank ** -0.5 * dt_scale
        self.dt_proj.weight.assign(
            jt.array(
                np.random.uniform(
                    -dt_std,
                    dt_std,
                    size=tuple(self.dt_proj.weight.shape),
                ).astype(np.float32)
            )
        )
        dt = np.exp(
            np.random.uniform(math.log(dt_min), math.log(dt_max), size=self.d_inner)
        )
        dt = np.maximum(dt, dt_init_floor).astype(np.float32)
        inverse_softplus = dt + np.log(-np.expm1(-dt))
        self.dt_proj.bias.assign(jt.array(inverse_softplus.astype(np.float32)))

        base = jt.arange(1, self.d_state + 1).float32().reshape(1, self.d_state)
        self.A_log = jt.log(base.broadcast((self.d_inner, self.d_state)))
        self.D = jt.ones((self.d_inner,), dtype="float32")

    def execute(self, hidden_states: jt.Var) -> jt.Var:
        if hidden_states.ndim != 3 or hidden_states.shape[2] != self.d_model:
            raise ValueError(
                f"Mamba expected (B,L,{self.d_model}), got {tuple(hidden_states.shape)}"
            )
        batch, length, _ = hidden_states.shape
        xz = self.in_proj(hidden_states).permute(0, 2, 1)
        x = xz[:, : self.d_inner, :]
        z = xz[:, self.d_inner :, :]

        # Conv1d padding=d_conv-1 followed by the historical left slice is the
        # same causal convolution used by mamba_ssm's reference path.
        x = self.conv1d(x)[:, :, :length]
        x = silu(x)

        projected = self.x_proj(x.permute(0, 2, 1))
        dt_part = projected[:, :, : self.dt_rank]
        b_part = projected[:, :, self.dt_rank : self.dt_rank + self.d_state]
        c_part = projected[:, :, self.dt_rank + self.d_state :]

        # mamba_ssm deliberately applies dt_proj without its bias here; the bias
        # is supplied separately to selective_scan before softplus.
        dt = jt.matmul(dt_part, self.dt_proj.weight.transpose(0, 1)).permute(0, 2, 1)
        b_var = b_part.permute(0, 2, 1).contiguous()
        c_var = c_part.permute(0, 2, 1).contiguous()
        a = -jt.exp(self.A_log)
        y = selective_scan(
            x.float32(),
            dt.float32(),
            a.float32(),
            b_var.float32(),
            c_var.float32(),
            self.D.float32(),
            z.float32(),
            self.dt_proj.bias.float32(),
        )
        return self.out_proj(y.permute(0, 2, 1))


class Block(nn.Module):
    def __init__(self, dim: int, layer_idx: int) -> None:
        super().__init__()
        self.mixer = Mamba(dim, layer_idx=layer_idx)
        self.norm = nn.LayerNorm(dim, eps=1e-5)

    def execute(self, hidden_states: jt.Var, residual: Optional[jt.Var] = None):
        residual = hidden_states if residual is None else hidden_states + residual
        hidden_states = self.norm(residual)
        hidden_states = self.mixer(hidden_states)
        return hidden_states, residual


class MixerModel(nn.Module):
    def __init__(self, d_model: int, n_layer: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([Block(d_model, index) for index in range(n_layer)])
        self.norm_f = nn.LayerNorm(d_model, eps=1e-5)

    def execute(self, hidden_states: jt.Var) -> jt.Var:
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)
        residual = hidden_states if residual is None else hidden_states + residual
        return self.norm_f(residual)
