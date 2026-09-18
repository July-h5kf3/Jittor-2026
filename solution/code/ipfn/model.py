"""Jittor inference port of the official IterativePFN feature network."""

from __future__ import annotations

import numpy as np
import jittor as jt
from jittor import nn


def gather_neighbors(x, idx):
    channels = x.shape[-1]
    return x.reindex(
        [idx.shape[0], idx.shape[1], idx.shape[2], channels],
        ["i0", "@e0(i0,i1,i2)", "i3"],
        extras=[idx],
    )


def knn_without_self(x, k):
    # Jittor's built-in KNN is hard-coded to xyz. Dynamic EdgeConv also needs
    # KNN in 16/48/144-D feature spaces. A fused scan keeps only K+1 minima
    # and avoids the previous O(N log N) full argsort bottleneck.
    keep = int(k) + 1
    distances = jt.empty((x.shape[0], x.shape[1], keep), dtype="float")
    indices = jt.empty((x.shape[0], x.shape[1], keep), dtype="int")
    source = r'''
__inline_static__
@python.jittor.auto_parallel(2, block_num=256)
void generic_knn(int batch, int batch_index, int count, int point_index,
                 int channels, const float *__restrict__ input,
                 float *__restrict__ distance, int *__restrict__ index) {
#define K %d
    input += batch_index * count * channels;
    distance += batch_index * count * K;
    index += batch_index * count * K;
    float best_distance[K];
    int best_index[K];
    #pragma unroll
    for (int slot=0; slot<K; ++slot) { best_distance[slot] = 1e30f; best_index[slot] = -1; }
    for (int candidate=0; candidate<count; ++candidate) {
        float value = 0.0f;
        for (int channel=0; channel<channels; ++channel) {
            float delta = input[point_index*channels+channel] - input[candidate*channels+channel];
            value += delta * delta;
        }
        int first = -1;
        #pragma unroll
        for (int slot=0; slot<K; ++slot) if (first == -1 && value < best_distance[slot]) first = slot;
        if (first == -1) continue;
        #pragma unroll
        for (int slot=K-1; slot>first; --slot) {
            best_distance[slot] = best_distance[slot-1];
            best_index[slot] = best_index[slot-1];
        }
        best_distance[first] = value;
        best_index[first] = candidate;
    }
    #pragma unroll
    for (int slot=0; slot<K; ++slot) {
        distance[point_index*K+slot] = best_distance[slot];
        index[point_index*K+slot] = best_index[slot];
    }
}
generic_knn(in0->shape[0], 0, in0->shape[1], 0, in0->shape[2], in0_p, out0_p, out1_p);
''' % keep
    _, indices = jt.code([x], [distances, indices], cpu_src=source, cuda_src=source)
    return indices[:, :, 1:keep]


def apply_bn(module, value):
    shape = value.shape
    flat = value.reshape((-1, shape[-1]))
    return module(flat).reshape(shape)


class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc1 = nn.Linear(2 * in_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.lin = nn.Linear(in_channels, out_channels)
        self.lin_bn = nn.BatchNorm1d(out_channels)

    def execute(self, x, idx):
        center = x.unsqueeze(2).broadcast(
            [x.shape[0], x.shape[1], idx.shape[-1], x.shape[-1]]
        )
        neighbor = gather_neighbors(x, idx)
        edge = jt.concat([center, neighbor - center], dim=-1)
        message = nn.relu(apply_bn(self.bn1, self.fc1(edge)))
        message = nn.relu(apply_bn(self.bn2, self.fc2(message)))
        aggregated = message.max(dim=2)
        residual = nn.relu(apply_bn(self.lin_bn, self.lin(x)))
        return aggregated + residual


class FeatureExtraction(nn.Module):
    def __init__(self, k=32):
        super().__init__()
        self.k = int(k)
        self.conv1 = EdgeConv(3, 16)
        self.conv2 = EdgeConv(16, 48)
        self.conv3 = EdgeConv(48, 144)
        self.conv4 = EdgeConv(16 + 48 + 144, 512)
        self.linear1 = nn.Linear(512, 256, bias=False)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 3)

    def execute(self, x):
        x1 = self.conv1(x, knn_without_self(x, self.k))
        x2 = self.conv2(x1, knn_without_self(x1, self.k))
        x3 = self.conv3(x2, knn_without_self(x2, self.k))
        combined = jt.concat([x1, x2, x3], dim=-1)
        x4 = self.conv4(combined, knn_without_self(x3, self.k))
        output = nn.relu(self.linear1(x4))
        output = nn.relu(self.linear2(output))
        return jt.tanh(self.linear3(output))


class IterativePFN(nn.Module):
    def __init__(self, num_modules=4, frame_knn=32):
        super().__init__()
        self.num_modules = int(num_modules)
        self.feature_nets = nn.ModuleList(
            [FeatureExtraction(k=frame_knn) for _ in range(self.num_modules)]
        )

    def execute(self, points, num_modules=None, return_stages=False):
        count = self.num_modules if num_modules is None else int(num_modules)
        current = points
        stages = []
        for index in range(count):
            current = current + self.feature_nets[index](current)
            stages.append(current)
        return stages if return_stages else current


def assign_linear(module, arrays, prefix):
    module.weight.assign(arrays[prefix + ".weight"])
    if getattr(module, "bias", None) is not None and prefix + ".bias" in arrays:
        module.bias.assign(arrays[prefix + ".bias"])


def assign_bn(module, arrays, prefix):
    module.weight.assign(arrays[prefix + ".weight"])
    module.bias.assign(arrays[prefix + ".bias"])
    module.running_mean.assign(arrays[prefix + ".running_mean"])
    module.running_var.assign(arrays[prefix + ".running_var"])


def assign_edge_conv(module, arrays, prefix):
    assign_linear(module.fc1, arrays, prefix + ".mlp.0")
    assign_bn(module.bn1, arrays, prefix + ".mlp.1")
    assign_linear(module.fc2, arrays, prefix + ".mlp.3")
    assign_bn(module.bn2, arrays, prefix + ".mlp.4")
    assign_linear(module.lin, arrays, prefix + ".lin.0")
    assign_bn(module.lin_bn, arrays, prefix + ".lin.1")


def load_converted_weights(model, path):
    arrays = np.load(path, allow_pickle=False)
    for module_index, feature in enumerate(model.feature_nets):
        prefix = "feature_nets.%d" % module_index
        assign_edge_conv(feature.conv1, arrays, prefix + ".conv1")
        assign_edge_conv(feature.conv2, arrays, prefix + ".conv2")
        assign_edge_conv(feature.conv3, arrays, prefix + ".conv3")
        assign_edge_conv(feature.conv4, arrays, prefix + ".conv4")
        assign_linear(feature.linear1, arrays, prefix + ".linear1")
        assign_linear(feature.linear2, arrays, prefix + ".linear2")
        assign_linear(feature.linear3, arrays, prefix + ".linear3")
    model.eval()
    return model


def export_converted_weights(model, path=None):
    """Save a trained Jittor model in the same NPZ schema used by inference."""
    arrays = {}

    def linear(module, prefix):
        arrays[prefix + ".weight"] = module.weight.numpy()
        if getattr(module, "bias", None) is not None:
            arrays[prefix + ".bias"] = module.bias.numpy()

    def bn(module, prefix):
        arrays[prefix + ".weight"] = module.weight.numpy()
        arrays[prefix + ".bias"] = module.bias.numpy()
        arrays[prefix + ".running_mean"] = module.running_mean.numpy()
        arrays[prefix + ".running_var"] = module.running_var.numpy()
        arrays[prefix + ".num_batches_tracked"] = np.asarray(0, dtype=np.int64)

    def edge(module, prefix):
        linear(module.fc1, prefix + ".mlp.0")
        bn(module.bn1, prefix + ".mlp.1")
        linear(module.fc2, prefix + ".mlp.3")
        bn(module.bn2, prefix + ".mlp.4")
        linear(module.lin, prefix + ".lin.0")
        bn(module.lin_bn, prefix + ".lin.1")

    for index, feature in enumerate(model.feature_nets):
        prefix = "feature_nets.%d" % index
        edge(feature.conv1, prefix + ".conv1")
        edge(feature.conv2, prefix + ".conv2")
        edge(feature.conv3, prefix + ".conv3")
        edge(feature.conv4, prefix + ".conv4")
        linear(feature.linear1, prefix + ".linear1")
        linear(feature.linear2, prefix + ".linear2")
        linear(feature.linear3, prefix + ".linear3")
    # Every MPI rank should materialise the variables, but only rank 0 should
    # pass a path and write the shared checkpoint.
    if path is not None:
        np.savez(path, **arrays)
    return arrays
