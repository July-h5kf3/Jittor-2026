"""Jittor-native selective scan used by Mamba.

The CUDA path is a Jittor custom operator with an explicit backward kernel.  It
has no PyTorch, Triton, ``mamba_ssm`` or ``causal_conv1d`` runtime dependency.
A small, fully differentiable Jittor reference path is retained for CPU tests.
"""

from __future__ import annotations

from typing import Tuple

import jittor as jt
from jittor import Function


_CHECKPOINT_INTERVAL = 16
_MAX_STATE = 32


def _softplus(value: jt.Var) -> jt.Var:
    # Stable softplus, equivalent to torch.nn.functional.softplus(beta=1).
    return jt.maximum(value, 0.0) + jt.log(1.0 + jt.exp(-jt.abs(value)))


def selective_scan_reference(
    u: jt.Var,
    delta: jt.Var,
    a: jt.Var,
    b_var: jt.Var,
    c_var: jt.Var,
    d_skip: jt.Var,
    z: jt.Var,
    delta_bias: jt.Var,
) -> jt.Var:
    """Slow Jittor reference implementation, intended for CPU/unit tests."""
    batch, channels, length = u.shape
    state_count = a.shape[1]
    state = jt.zeros((batch, channels, state_count), dtype=u.dtype)
    outputs = []
    a_batch = a.unsqueeze(0)
    d_batch = d_skip.reshape(1, channels)
    bias = delta_bias.reshape(1, channels)
    for step in range(length):
        dt = _softplus(delta[:, :, step] + bias)
        transition = jt.exp(dt.unsqueeze(-1) * a_batch)
        drive = (
            dt.unsqueeze(-1)
            * b_var[:, :, step].unsqueeze(1)
            * u[:, :, step].unsqueeze(-1)
        )
        state = transition * state + drive
        value = (state * c_var[:, :, step].unsqueeze(1)).sum(dim=-1)
        value = value + d_batch * u[:, :, step]
        gate_input = z[:, :, step]
        value = value * gate_input * jt.sigmoid(gate_input)
        outputs.append(value.unsqueeze(-1))
    return jt.concat(outputs, dim=-1)


_FORWARD_HEADER = r"""
@alias(u,in0)
@alias(delta,in1)
@alias(a,in2)
@alias(b_var,in3)
@alias(c_var,in4)
@alias(d_skip,in5)
@alias(z,in6)
@alias(delta_bias,in7)
@alias(output,out0)
@alias(checkpoints,out1)
"""

_BACKWARD_HEADER = r"""
@alias(u,in0)
@alias(delta,in1)
@alias(a,in2)
@alias(b_var,in3)
@alias(c_var,in4)
@alias(d_skip,in5)
@alias(z,in6)
@alias(delta_bias,in7)
@alias(checkpoints,in8)
@alias(grad_output,in9)
@alias(grad_u,out0)
@alias(grad_delta,out1)
@alias(grad_a,out2)
@alias(grad_b,out3)
@alias(grad_c,out4)
@alias(grad_d,out5)
@alias(grad_z,out6)
@alias(grad_delta_bias,out7)
"""

_FORWARD_CUDA = r"""

__device__ __forceinline__ float plr_softplus(float x) {
    return x > 20.0f ? x : log1pf(expf(x));
}

__device__ __forceinline__ float plr_silu(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void plr_selective_scan_forward(
    const float* u,
    const float* delta,
    const float* A,
    const float* Bv,
    const float* Cv,
    const float* Dskip,
    const float* z,
    const float* delta_bias,
    float* output,
    float* checkpoints,
    int batch,
    int channels,
    int length,
    int states,
    int checkpoint_count
) {
    int bd = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels;
    if (bd >= total) return;
    int b = bd / channels;
    int d = bd - b * channels;
    if (states > 32) return;

    float state[32];
    #pragma unroll
    for (int n = 0; n < 32; ++n) state[n] = 0.0f;
    for (int n = 0; n < states; ++n) {
        checkpoints[((bd * checkpoint_count) * states) + n] = 0.0f;
    }

    int block_index = 0;
    for (int t = 0; t < length; ++t) {
        int ud = (bd * length) + t;
        float dt = plr_softplus(delta[ud] + delta_bias[d]);
        float input = u[ud];
        float value = Dskip[d] * input;
        for (int n = 0; n < states; ++n) {
            int an = d * states + n;
            int bn = (b * states + n) * length + t;
            float transition = expf(dt * A[an]);
            state[n] = transition * state[n] + dt * Bv[bn] * input;
            value += state[n] * Cv[bn];
        }
        output[ud] = value * plr_silu(z[ud]);
        if (((t + 1) % 16) == 0 || t + 1 == length) {
            ++block_index;
            for (int n = 0; n < states; ++n) {
                checkpoints[((bd * checkpoint_count + block_index) * states) + n] = state[n];
            }
        }
    }
}

int batch = in0_shape0;
int channels = in0_shape1;
int length = in0_shape2;
int states = in2_shape1;
int checkpoint_count = out1_shape2;
int total = batch * channels;
plr_selective_scan_forward<<<(total + 127) / 128, 128>>>(
    in0_p, in1_p, in2_p, in3_p, in4_p, in5_p, in6_p, in7_p,
    out0_p, out1_p, batch, channels, length, states, checkpoint_count
);
"""


_BACKWARD_CUDA = r"""

__device__ __forceinline__ float plr_softplus_bwd(float x) {
    return x > 20.0f ? x : log1pf(expf(x));
}

__device__ __forceinline__ float plr_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void plr_zero(float* output, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = index; i < count; i += blockDim.x * gridDim.x) output[i] = 0.0f;
}

__global__ void plr_selective_scan_backward(
    const float* u,
    const float* delta,
    const float* A,
    const float* Bv,
    const float* Cv,
    const float* Dskip,
    const float* z,
    const float* delta_bias,
    const float* checkpoints,
    const float* grad_output,
    float* grad_u,
    float* grad_delta,
    float* grad_A,
    float* grad_B,
    float* grad_C,
    float* grad_D,
    float* grad_z,
    float* grad_delta_bias,
    int batch,
    int channels,
    int length,
    int states,
    int checkpoint_count
) {
    int bd = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels;
    if (bd >= total || states > 32) return;
    int b = bd / channels;
    int d = bd - b * channels;
    int block_count = checkpoint_count - 1;

    float adjoint[32];
    float grad_a_local[32];
    float block_states[17 * 32];
    #pragma unroll
    for (int n = 0; n < 32; ++n) {
        adjoint[n] = 0.0f;
        grad_a_local[n] = 0.0f;
    }
    float grad_d_local = 0.0f;
    float grad_bias_local = 0.0f;

    for (int block = block_count - 1; block >= 0; --block) {
        int start = block * 16;
        int end = start + 16;
        if (end > length) end = length;
        int block_length = end - start;

        for (int n = 0; n < states; ++n) {
            block_states[n] = checkpoints[((bd * checkpoint_count + block) * states) + n];
        }

        for (int i = 0; i < block_length; ++i) {
            int t = start + i;
            int ud = bd * length + t;
            float dt = plr_softplus_bwd(delta[ud] + delta_bias[d]);
            float input = u[ud];
            for (int n = 0; n < states; ++n) {
                int an = d * states + n;
                int bn = (b * states + n) * length + t;
                float previous = block_states[i * 32 + n];
                float transition = expf(dt * A[an]);
                block_states[(i + 1) * 32 + n] =
                    transition * previous + dt * Bv[bn] * input;
            }
        }

        for (int i = block_length - 1; i >= 0; --i) {
            int t = start + i;
            int ud = bd * length + t;
            float raw_dt = delta[ud] + delta_bias[d];
            float dt = plr_softplus_bwd(raw_dt);
            float input = u[ud];
            float y_before_gate = Dskip[d] * input;
            for (int n = 0; n < states; ++n) {
                int bn = (b * states + n) * length + t;
                y_before_gate += block_states[(i + 1) * 32 + n] * Cv[bn];
            }

            float sigmoid_z = plr_sigmoid(z[ud]);
            float silu_z = z[ud] * sigmoid_z;
            float upstream = grad_output[ud];
            float grad_y = upstream * silu_z;
            grad_z[ud] = upstream * y_before_gate * sigmoid_z *
                (1.0f + z[ud] * (1.0f - sigmoid_z));

            float grad_input = grad_y * Dskip[d];
            float grad_dt = 0.0f;
            grad_d_local += grad_y * input;

            for (int n = 0; n < states; ++n) {
                int an = d * states + n;
                int bn = (b * states + n) * length + t;
                float previous = block_states[i * 32 + n];
                float current = block_states[(i + 1) * 32 + n];
                float transition = expf(dt * A[an]);
                float grad_state = adjoint[n] + grad_y * Cv[bn];

                atomicAdd(&grad_C[bn], grad_y * current);
                float grad_transition = grad_state * previous;
                grad_dt += grad_transition * transition * A[an];
                grad_dt += grad_state * Bv[bn] * input;
                grad_a_local[n] += grad_transition * transition * dt;
                atomicAdd(&grad_B[bn], grad_state * dt * input);
                grad_input += grad_state * dt * Bv[bn];
                adjoint[n] = grad_state * transition;
            }

            float sigmoid_dt = plr_sigmoid(raw_dt);
            float grad_raw_dt = grad_dt * sigmoid_dt;
            grad_u[ud] = grad_input;
            grad_delta[ud] = grad_raw_dt;
            grad_bias_local += grad_raw_dt;
        }
    }

    for (int n = 0; n < states; ++n) {
        atomicAdd(&grad_A[d * states + n], grad_a_local[n]);
    }
    atomicAdd(&grad_D[d], grad_d_local);
    atomicAdd(&grad_delta_bias[d], grad_bias_local);
}

int count_u = in0_shape0 * in0_shape1 * in0_shape2;
int count_a = in2_shape0 * in2_shape1;
int count_bc = in3_shape0 * in3_shape1 * in3_shape2;
int count_d = in5_shape0;
plr_zero<<<128, 256>>>(out0_p, count_u);
plr_zero<<<128, 256>>>(out1_p, count_u);
plr_zero<<<128, 256>>>(out2_p, count_a);
plr_zero<<<128, 256>>>(out3_p, count_bc);
plr_zero<<<128, 256>>>(out4_p, count_bc);
plr_zero<<<128, 256>>>(out5_p, count_d);
plr_zero<<<128, 256>>>(out6_p, count_u);
plr_zero<<<128, 256>>>(out7_p, count_d);

int batch = in0_shape0;
int channels = in0_shape1;
int length = in0_shape2;
int states = in2_shape1;
int checkpoint_count = in8_shape2;
int total = batch * channels;
plr_selective_scan_backward<<<(total + 63) / 64, 64>>>(
    in0_p, in1_p, in2_p, in3_p, in4_p, in5_p, in6_p, in7_p,
    in8_p, in9_p,
    out0_p, out1_p, out2_p, out3_p, out4_p, out5_p, out6_p, out7_p,
    batch, channels, length, states, checkpoint_count
);
"""


class _SelectiveScanCUDA(Function):
    def execute(
        self,
        u: jt.Var,
        delta: jt.Var,
        a: jt.Var,
        b_var: jt.Var,
        c_var: jt.Var,
        d_skip: jt.Var,
        z: jt.Var,
        delta_bias: jt.Var,
    ) -> jt.Var:
        batch, channels, length = u.shape
        states = a.shape[1]
        checkpoint_count = (length + _CHECKPOINT_INTERVAL - 1) // _CHECKPOINT_INTERVAL + 1
        output, checkpoints = jt.code(
            [u.shape, (batch, channels, checkpoint_count, states)],
            [u.dtype, u.dtype],
            [u, delta, a, b_var, c_var, d_skip, z, delta_bias],
            cuda_header=_FORWARD_HEADER,
            cuda_src=_FORWARD_CUDA,
        )
        self.saved = (u, delta, a, b_var, c_var, d_skip, z, delta_bias, checkpoints)
        return output

    def grad(self, grad_output: jt.Var) -> Tuple[jt.Var, ...]:
        u, delta, a, b_var, c_var, d_skip, z, delta_bias, checkpoints = self.saved
        return tuple(
            jt.code(
                [
                    u.shape,
                    delta.shape,
                    a.shape,
                    b_var.shape,
                    c_var.shape,
                    d_skip.shape,
                    z.shape,
                    delta_bias.shape,
                ],
                [
                    u.dtype,
                    delta.dtype,
                    a.dtype,
                    b_var.dtype,
                    c_var.dtype,
                    d_skip.dtype,
                    z.dtype,
                    delta_bias.dtype,
                ],
                [
                    u,
                    delta,
                    a,
                    b_var,
                    c_var,
                    d_skip,
                    z,
                    delta_bias,
                    checkpoints,
                    grad_output,
                ],
                cuda_header=_BACKWARD_HEADER,
                cuda_src=_BACKWARD_CUDA,
            )
        )


def _validate_inputs(
    u: jt.Var,
    delta: jt.Var,
    a: jt.Var,
    b_var: jt.Var,
    c_var: jt.Var,
    d_skip: jt.Var,
    z: jt.Var,
    delta_bias: jt.Var,
) -> None:
    if u.ndim != 3 or delta.shape != u.shape or z.shape != u.shape:
        raise ValueError("u, delta and z must share shape (B,D,L)")
    batch, channels, length = u.shape
    if a.ndim != 2 or a.shape[0] != channels:
        raise ValueError("A must have shape (D,N)")
    states = a.shape[1]
    if states <= 0 or states > _MAX_STATE:
        raise ValueError(f"selective scan supports 1..{_MAX_STATE} states")
    if b_var.shape != (batch, states, length) or c_var.shape != b_var.shape:
        raise ValueError("B and C must have shape (B,N,L)")
    if d_skip.shape != (channels,) or delta_bias.shape != (channels,):
        raise ValueError("D and delta_bias must have shape (D,)")
    for value in (u, delta, a, b_var, c_var, d_skip, z, delta_bias):
        if value.dtype != "float32":
            raise TypeError("the PLR selective scan requires float32 tensors")


def selective_scan(
    u: jt.Var,
    delta: jt.Var,
    a: jt.Var,
    b_var: jt.Var,
    c_var: jt.Var,
    d_skip: jt.Var,
    z: jt.Var,
    delta_bias: jt.Var,
) -> jt.Var:
    """Run the Mamba selective scan with SiLU gating and softplus delta."""
    _validate_inputs(u, delta, a, b_var, c_var, d_skip, z, delta_bias)
    if bool(jt.flags.use_cuda):
        return _SelectiveScanCUDA.apply(
            u.contiguous(),
            delta.contiguous(),
            a.contiguous(),
            b_var.contiguous(),
            c_var.contiguous(),
            d_skip.contiguous(),
            z.contiguous(),
            delta_bias.contiguous(),
        )
    return selective_scan_reference(u, delta, a, b_var, c_var, d_skip, z, delta_bias)
