"""Small, numerical compatibility checks for the Jittor ACL execution path."""

from __future__ import annotations

import tempfile
import unittest

import numpy as np

try:
    import jittor as jt
    from jittor import nn, optim
except ImportError:  # Keep the normal local test suite runnable without Jittor.
    jt = None
    nn = optim = None

from plr3d.backend import configure_device


RTOL = 1e-4
ATOL = 1e-5
HAS_ACL = bool(jt is not None and getattr(jt.compiler, "has_acl", False))


def _conv1d_reference(values, weight, bias):
    """Reference cross-correlation for the fixed no-padding test inputs."""
    batch, _, length = values.shape
    out_channels, _, kernel = weight.shape
    output = np.empty((batch, out_channels, length - kernel + 1), dtype=np.float32)
    groups = values.shape[1] // weight.shape[1]
    out_per_group = out_channels // groups
    for batch_index in range(batch):
        for out_channel in range(out_channels):
            group = out_channel // out_per_group
            input_start = group * weight.shape[1]
            for position in range(output.shape[2]):
                window = values[
                    batch_index,
                    input_start : input_start + weight.shape[1],
                    position : position + kernel,
                ]
                output[batch_index, out_channel, position] = (
                    (window * weight[out_channel]).sum() + bias[out_channel]
                )
    return output


@unittest.skipUnless(HAS_ACL, "Jittor ACL backend is unavailable")
class ACLPrimitiveCompatibilityTests(unittest.TestCase):
    """Run every operation after selecting ACL, never by silently using CPU."""

    @classmethod
    def setUpClass(cls):
        configure_device("acl", jt)
        jt.set_global_seed(20260810)
        cls._assert_acl()

    @classmethod
    def _assert_acl(cls):
        if int(jt.flags.use_acl) != 1:
            raise AssertionError("ACL was disabled during primitive test")

    def setUp(self):
        self._assert_acl()

    def tearDown(self):
        jt.sync_all()
        self._assert_acl()

    def assertArrayClose(self, actual, expected, *, rtol=RTOL, atol=ATOL):
        np.testing.assert_allclose(actual.numpy(), expected, rtol=rtol, atol=atol)

    def test_matmul_and_bmm_fp32(self):
        left_np = np.arange(6, dtype=np.float32).reshape(2, 3) / 5.0
        right_np = np.arange(12, dtype=np.float32).reshape(3, 4) / 7.0
        left = jt.array(left_np)
        matrix = jt.matmul(left, jt.array(right_np))
        self.assertEqual(tuple(matrix.shape), (2, 4))
        self.assertArrayClose(matrix, left_np @ right_np)

        batch_left_np = np.arange(12, dtype=np.float32).reshape(2, 2, 3) / 9.0
        batch_right_np = np.arange(12, dtype=np.float32).reshape(2, 3, 2) / 11.0
        batch_left = jt.array(batch_left_np)
        batch = jt.matmul(batch_left, jt.array(batch_right_np))
        self.assertEqual(tuple(batch.shape), (2, 2, 2))
        self.assertArrayClose(batch, np.matmul(batch_left_np, batch_right_np))

        gradient = jt.grad(batch.sum(), batch_left)
        expected_gradient = np.broadcast_to(
            batch_right_np.sum(axis=2)[:, np.newaxis, :], batch_left_np.shape
        )
        self.assertArrayClose(gradient, expected_gradient)

    def test_linear_conv1d_and_depthwise_conv1d(self):
        linear_input_np = np.array([[1.0, -2.0, 0.5], [0.0, 3.0, -1.0]], dtype=np.float32)
        linear_weight_np = np.array([[1.0, 2.0, -1.0], [-0.5, 1.5, 0.25]], dtype=np.float32)
        linear_bias_np = np.array([0.25, -0.75], dtype=np.float32)
        linear = nn.Linear(3, 2)
        linear.weight.assign(jt.array(linear_weight_np))
        linear.bias.assign(jt.array(linear_bias_np))
        self.assertArrayClose(
            linear(jt.array(linear_input_np)),
            linear_input_np @ linear_weight_np.T + linear_bias_np,
        )

        input_np = np.array(
            [[[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 2.0, 1.5]]], dtype=np.float32
        )
        conv_weight_np = np.array(
            [[[1.0, -1.0], [0.5, 0.25]], [[-0.5, 1.0], [1.0, 0.0]]], dtype=np.float32
        )
        conv_bias_np = np.array([0.25, -0.5], dtype=np.float32)
        conv = nn.Conv1d(2, 2, 2, bias=True)
        conv.weight.assign(jt.array(conv_weight_np))
        conv.bias.assign(jt.array(conv_bias_np))
        self.assertArrayClose(
            conv(jt.array(input_np)), _conv1d_reference(input_np, conv_weight_np, conv_bias_np)
        )

        depthwise_weight_np = np.array([[[1.0, 0.5]], [[-1.0, 2.0]]], dtype=np.float32)
        depthwise_bias_np = np.array([0.0, 0.25], dtype=np.float32)
        depthwise = nn.Conv1d(2, 2, 2, groups=2, bias=True)
        depthwise.weight.assign(jt.array(depthwise_weight_np))
        depthwise.bias.assign(jt.array(depthwise_bias_np))
        self.assertArrayClose(
            depthwise(jt.array(input_np)),
            _conv1d_reference(input_np, depthwise_weight_np, depthwise_bias_np),
        )

    def test_batchnorm_layernorm_and_activations(self):
        values_np = np.array(
            [[[1.0, 3.0], [2.0, -1.0]], [[5.0, 7.0], [4.0, 0.0]]], dtype=np.float32
        )
        batchnorm = nn.BatchNorm1d(2, eps=1e-5, is_train=False)
        batchnorm.weight.assign(jt.array(np.array([1.5, -0.5], dtype=np.float32)))
        batchnorm.bias.assign(jt.array(np.array([0.25, 1.0], dtype=np.float32)))
        batchnorm.running_mean.assign(jt.array(np.array([2.0, 0.5], dtype=np.float32)))
        batchnorm.running_var.assign(jt.array(np.array([4.0, 9.0], dtype=np.float32)))
        batchnorm_expected = (
            (values_np - np.array([2.0, 0.5], dtype=np.float32).reshape(1, 2, 1))
            / np.sqrt(np.array([4.0, 9.0], dtype=np.float32).reshape(1, 2, 1) + 1e-5)
            * np.array([1.5, -0.5], dtype=np.float32).reshape(1, 2, 1)
            + np.array([0.25, 1.0], dtype=np.float32).reshape(1, 2, 1)
        )
        self.assertArrayClose(batchnorm(jt.array(values_np)), batchnorm_expected)

        layer_input_np = np.array([[1.0, 2.0, 4.0], [-2.0, 0.0, 1.0]], dtype=np.float32)
        layernorm = nn.LayerNorm(3, eps=1e-5)
        layer_weight_np = np.array([1.0, 0.5, -1.0], dtype=np.float32)
        layer_bias_np = np.array([0.25, -0.5, 1.0], dtype=np.float32)
        layernorm.weight.assign(jt.array(layer_weight_np))
        layernorm.bias.assign(jt.array(layer_bias_np))
        mean = layer_input_np.mean(axis=-1, keepdims=True)
        variance = ((layer_input_np - mean) ** 2).mean(axis=-1, keepdims=True)
        layer_expected = (layer_input_np - mean) / np.sqrt(variance + 1e-5)
        layer_expected = layer_expected * layer_weight_np + layer_bias_np
        self.assertArrayClose(layernorm(jt.array(layer_input_np)), layer_expected)

        activation_input_np = np.array([-2.0, -0.5, 0.0, 1.5], dtype=np.float32)
        activation_input = jt.array(activation_input_np)
        self.assertArrayClose(nn.relu(activation_input), np.maximum(activation_input_np, 0.0))
        self.assertArrayClose(
            activation_input * jt.sigmoid(activation_input),
            activation_input_np / (1.0 + np.exp(-activation_input_np)),
        )

    def test_gather_scatter_argmax_and_topk(self):
        from plr3d.ops.geometry import gather_points

        points_np = np.arange(24, dtype=np.float32).reshape(2, 4, 3) / 10.0
        indices_np = np.array([[[3, 1], [0, 2]], [[2, 0], [1, 3]]], dtype=np.int32)
        gathered = gather_points(jt.array(points_np), jt.array(indices_np))
        expected_gathered = np.stack(
            [points_np[batch_index][indices_np[batch_index]] for batch_index in range(2)]
        )
        self.assertEqual(tuple(gathered.shape), (2, 2, 2, 3))
        self.assertArrayClose(gathered, expected_gathered)

        scatter_indices_np = np.array([[0, 0], [2, 2], [1, 1], [2, 2]], dtype=np.int32)
        scatter_values_np = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [-1.0, 1.0]], dtype=np.float32)
        scattered = jt.zeros((4, 2), dtype="float32").scatter_(
            0, jt.array(scatter_indices_np), jt.array(scatter_values_np), reduce="add"
        )
        expected_scatter = np.array([[1.0, 2.0], [5.0, 6.0], [2.0, 5.0], [0.0, 0.0]], dtype=np.float32)
        self.assertArrayClose(scattered, expected_scatter)

        ranking_np = np.array([[0.1, 0.7, 0.3, 0.5], [1.0, -2.0, 4.0, 3.0]], dtype=np.float32)
        ranking = jt.array(ranking_np)
        argmax_indices, argmax_values = jt.argmax(ranking, dim=1)
        self.assertEqual(tuple(argmax_indices.shape), (2,))
        expected_argmax_indices = np.argmax(ranking_np, axis=1).astype(np.int32)
        np.testing.assert_array_equal(argmax_indices.numpy(), expected_argmax_indices)
        self.assertArrayClose(
            argmax_values, np.take_along_axis(ranking_np, expected_argmax_indices[:, np.newaxis], axis=1).reshape(-1)
        )
        top_values, top_indices = jt.topk(ranking, k=2, dim=1, largest=True)
        expected_indices = np.argsort(-ranking_np, axis=1)[:, :2].astype(np.int32)
        self.assertArrayClose(top_values, np.take_along_axis(ranking_np, expected_indices, axis=1))
        np.testing.assert_array_equal(top_indices.numpy(), expected_indices)

    def test_coordinate_and_feature_knn(self):
        from plr3d.ops.geometry import knn_indices, knn_points
        from rot_jittor.src.model.feature import get_knn_idx

        query_np = np.array(
            [[[0.0, 0.0, 0.0], [2.1, 0.0, 0.0], [0.0, 2.2, 0.0]]], dtype=np.float32
        )
        reference_np = np.array(
            [[[0.2, 0.0, 0.0], [2.0, 0.1, 0.0], [0.0, 2.0, 0.3], [4.0, 4.0, 4.0]]],
            dtype=np.float32,
        )
        coordinate_distance, coordinate_indices, neighbours = knn_points(
            jt.array(query_np), jt.array(reference_np), k=2
        )
        coordinate_pairwise = ((query_np[:, :, np.newaxis] - reference_np[:, np.newaxis]) ** 2).sum(axis=-1)
        coordinate_expected_indices = np.argsort(coordinate_pairwise, axis=-1)[:, :, :2].astype(np.int32)
        coordinate_expected_distances = np.take_along_axis(
            coordinate_pairwise, coordinate_expected_indices, axis=-1
        )
        self.assertArrayClose(coordinate_distance, coordinate_expected_distances)
        np.testing.assert_array_equal(coordinate_indices.numpy(), coordinate_expected_indices)
        self.assertArrayClose(
            neighbours, np.take_along_axis(
                reference_np[:, np.newaxis],
                coordinate_expected_indices[:, :, :, np.newaxis],
                axis=2,
            )
        )

        feature_np = np.array(
            [[[0.0, 0.0], [1.0, 2.0], [2.5, 0.5], [4.0, 3.0]]], dtype=np.float32
        )
        feature_pairwise = ((feature_np[:, :, np.newaxis] - feature_np[:, np.newaxis]) ** 2).sum(axis=-1)
        feature_expected_indices = np.argsort(feature_pairwise, axis=-1)[:, :, :2].astype(np.int32)
        feature_distances, feature_indices = knn_indices(jt.array(feature_np), jt.array(feature_np), k=2)
        self.assertArrayClose(feature_distances, np.take_along_axis(feature_pairwise, feature_expected_indices, axis=-1))
        np.testing.assert_array_equal(feature_indices.numpy(), feature_expected_indices)
        rot_indices = get_knn_idx(jt.array(feature_np), jt.array(feature_np), k=1, offset=1)
        np.testing.assert_array_equal(rot_indices.numpy(), feature_expected_indices[:, :, 1:])

    def test_optimizer_step_and_checkpoint_roundtrip(self):
        inputs_np = np.array([[1.0, 2.0], [-1.0, 0.5], [3.0, -2.0]], dtype=np.float32)
        targets_np = np.array([[0.5], [-1.0], [2.0]], dtype=np.float32)
        model = nn.Linear(2, 1)
        model.weight.assign(jt.array(np.array([[0.25, -0.5]], dtype=np.float32)))
        model.bias.assign(jt.array(np.array([0.1], dtype=np.float32)))
        optimizer = optim.SGD(model.parameters(), lr=0.1)

        inputs = jt.array(inputs_np)
        targets = jt.array(targets_np)
        loss = ((model(inputs) - targets) ** 2).mean()
        self.assertTrue(np.isfinite(loss.numpy()).all())
        before = model.weight.numpy().copy()
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()
        jt.sync_all()
        after = model.weight.numpy()
        self.assertFalse(np.allclose(before, after))
        self.assertTrue(np.isfinite(after).all())

        expected_output = model(inputs)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = f"{directory}/linear.pkl"
            jt.save({"state_dict": model.state_dict()}, checkpoint_path)
            restored = nn.Linear(2, 1)
            restored.load_parameters(jt.load(checkpoint_path)["state_dict"])
            self.assertArrayClose(restored(inputs), expected_output.numpy())


if __name__ == "__main__":
    unittest.main()
