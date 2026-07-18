from pathlib import Path
import unittest

import jittor as jt
import numpy as np
from omegaconf import OmegaConf

from src.model.straightpcf import VelocityNet


ROOT = Path(__file__).resolve().parents[1]


class VelocityConditionTests(unittest.TestCase):
    def test_zero_initialized_condition_preserves_checkpoint_function(self):
        jt.flags.use_cuda = 0
        baseline = VelocityNet(
            8,
            32,
            16,
            decoder_type="graph",
            film=True,
        )
        conditioned = VelocityNet(
            8,
            32,
            16,
            decoder_type="graph",
            film=True,
            condition_dim=2,
        )
        conditioned.load_state_dict(baseline.state_dict())
        baseline.eval()
        conditioned.eval()

        rng = np.random.RandomState(31)
        points = jt.array(rng.randn(2, 17, 3).astype(np.float32))
        condition = jt.array(rng.rand(2, 2).astype(np.float32))
        baseline_output = baseline(points).numpy()
        conditioned_output = conditioned(points, condition=condition).numpy()
        np.testing.assert_array_equal(conditioned_output, baseline_output)

    def test_cond001_configs_isolate_time_stage_condition(self):
        cvm = OmegaConf.load(
            ROOT / "configs" / "model" / "spcfgfncond001_cvm.yaml"
        )
        spcf = OmegaConf.load(
            ROOT / "configs" / "model" / "spcfgfncond001_spcf.yaml"
        )
        control = OmegaConf.load(
            ROOT / "configs" / "model" / "spcfgfncvm002_cvm.yaml"
        )
        self.assertEqual(cvm.cvm_condition, "time_stage")
        self.assertEqual(spcf.cvm_condition, "time_stage")
        self.assertNotIn("cvm_condition", control)

        for name in (
            "train_spcfgfncond001_cvm.yaml",
            "train_spcfgfncond001.yaml",
            "predict_spcfgfncond001a105_local2.yaml",
        ):
            task = OmegaConf.load(ROOT / "configs" / "task" / name)
            for kind in ("data", "transform", "system", "model"):
                component = task.components[kind]
                self.assertTrue(
                    (ROOT / "configs" / kind / f"{component}.yaml").is_file()
                )


if __name__ == "__main__":
    unittest.main()
