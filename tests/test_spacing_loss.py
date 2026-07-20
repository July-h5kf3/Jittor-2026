from pathlib import Path
import unittest

import jittor as jt
import numpy as np
from omegaconf import OmegaConf

from src.model.straightpcf import StraightPCFModule


ROOT = Path(__file__).resolve().parents[1]


class LocalSpacingLossTests(unittest.TestCase):
    def _model(self, **overrides):
        cfg = OmegaConf.load(ROOT / "configs/model/spcfgfncvm002_cvm.yaml")
        cfg.lam_spacing = overrides.get("lam_spacing", 1.0)
        cfg.spacing_k = overrides.get("spacing_k", 4)
        cfg.spacing_points = overrides.get("spacing_points", 16)
        transform = OmegaConf.load(ROOT / "configs/transform/spcf_n_rot001.yaml")
        return StraightPCFModule(cfg, transform)

    def test_exact_geometry_has_zero_spacing_loss(self):
        jt.flags.use_cuda = 0
        model = self._model()
        clean = jt.array(
            np.random.RandomState(41).randn(2, 17, 3).astype(np.float32)
        )
        loss = float(model._local_spacing_loss(clean, clean).item())
        self.assertLess(loss, 1e-10)

    def test_contraction_and_expansion_are_both_penalized(self):
        jt.flags.use_cuda = 0
        model = self._model()
        clean = jt.array(
            np.random.RandomState(43).randn(2, 17, 3).astype(np.float32)
        )
        contracted = clean * 0.8
        expanded = clean * 1.2
        contracted_loss = float(
            model._local_spacing_loss(contracted, clean).item()
        )
        expanded_loss = float(model._local_spacing_loss(expanded, clean).item())
        self.assertGreater(contracted_loss, 0.0)
        self.assertGreater(expanded_loss, 0.0)

    def test_default_best_config_keeps_spacing_loss_disabled(self):
        cfg = OmegaConf.load(ROOT / "configs/model/spcfgfncvm002_cvm.yaml")
        self.assertNotIn("lam_spacing", cfg)
        model = self._model(lam_spacing=0.0)
        self.assertEqual(model.lam_spacing, 0.0)

    def test_uni001_configs_isolate_the_spacing_objective(self):
        control = OmegaConf.load(ROOT / "configs/model/spcfgfncvm002_cvm.yaml")
        candidate = OmegaConf.load(ROOT / "configs/model/spcfgfnuni001_cvm.yaml")
        self.assertNotIn("lam_spacing", control)
        self.assertEqual(candidate.lam_spacing, 0.50)
        self.assertEqual(candidate.spacing_k, 8)
        self.assertEqual(candidate.spacing_points, 256)
        for name in (
            "train_spcfgfnuni001_cvm.yaml",
            "train_spcfgfnuni001.yaml",
            "predict_spcfgfnuni001a105_local2.yaml",
            "predict_spcfgfnuni001a105_local2_pass2.yaml",
        ):
            task = OmegaConf.load(ROOT / "configs/task" / name)
            for kind in ("data", "transform", "system", "model"):
                component = task.components[kind]
                self.assertTrue(
                    (ROOT / "configs" / kind / f"{component}.yaml").is_file()
                )

    def test_loss_only_candidate_preserves_inference_function(self):
        jt.flags.use_cuda = 0
        control_cfg = OmegaConf.load(ROOT / "configs/model/spcfgfncvm002_cvm.yaml")
        candidate_cfg = OmegaConf.load(ROOT / "configs/model/spcfgfnuni001_cvm.yaml")
        transform = OmegaConf.load(ROOT / "configs/transform/spcf_n_rot001.yaml")
        control = StraightPCFModule(control_cfg, transform)
        candidate = StraightPCFModule(candidate_cfg, transform)
        candidate.load_state_dict(control.state_dict())
        control.eval()
        candidate.eval()
        points = jt.array(
            np.random.RandomState(47).randn(1, 40, 3).astype(np.float32)
        )
        baseline = control.velocity_nets[0](points).numpy()
        proposed = candidate.velocity_nets[0](points).numpy()
        np.testing.assert_array_equal(proposed, baseline)


if __name__ == "__main__":
    unittest.main()
