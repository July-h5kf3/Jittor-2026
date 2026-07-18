from pathlib import Path
import unittest

import jittor as jt
import numpy as np
from omegaconf import OmegaConf

from src.model.feature import PointNeXtLiteHierarchy
from src.model.straightpcf import VelocityNet


ROOT = Path(__file__).resolve().parents[1]


class PointNeXtLiteTests(unittest.TestCase):
    def test_zero_initialized_branch_is_an_exact_identity(self):
        jt.flags.use_cuda = 0
        rng = np.random.RandomState(17)
        x = jt.array(rng.randn(2, 17, 32).astype(np.float32))
        pos = jt.array(rng.randn(2, 17, 3).astype(np.float32))
        module = PointNeXtLiteHierarchy(
            dim=32,
            hidden_dim=16,
            stride=4,
            k=4,
            interpolation_k=3,
        )
        output = module(x, pos)
        np.testing.assert_array_equal(output.numpy(), x.numpy())

    def test_configs_isolate_encoder_type(self):
        cvm = OmegaConf.load(ROOT / "configs" / "model" / "spcfgfnpnx001_cvm.yaml")
        spcf = OmegaConf.load(ROOT / "configs" / "model" / "spcfgfnpnx001_spcf.yaml")
        control = OmegaConf.load(ROOT / "configs" / "model" / "spcfgfncvm002_cvm.yaml")
        self.assertEqual(cvm.encoder_type, "pointnext_lite")
        self.assertEqual(spcf.encoder_type, "pointnext_lite")
        self.assertNotIn("encoder_type", control)

        for name in (
            "train_spcfgfnpnx001_cvm.yaml",
            "train_spcfgfnpnx001.yaml",
            "predict_spcfgfnpnx001a105_local2.yaml",
        ):
            task = OmegaConf.load(ROOT / "configs" / "task" / name)
            for kind in ("data", "transform", "system", "model"):
                component = task.components[kind]
                self.assertTrue(
                    (ROOT / "configs" / kind / f"{component}.yaml").is_file()
                )

    def test_edgeconv_state_initializes_identical_pointnext_predictions(self):
        jt.flags.use_cuda = 0
        baseline = VelocityNet(
            8,
            32,
            16,
            decoder_type="graph",
            film=True,
        )
        pointnext = VelocityNet(
            8,
            32,
            16,
            decoder_type="graph",
            film=True,
            encoder_type="pointnext_lite",
            hierarchy_hidden_dim=16,
            hierarchy_stride=4,
            hierarchy_k=4,
        )
        pointnext.load_state_dict(baseline.state_dict())
        baseline.eval()
        pointnext.eval()

        points = jt.array(
            np.random.RandomState(23).randn(1, 17, 3).astype(np.float32)
        )
        baseline_output = baseline(points).numpy()
        pointnext_output = pointnext(points).numpy()
        np.testing.assert_array_equal(pointnext_output, baseline_output)


if __name__ == "__main__":
    unittest.main()
