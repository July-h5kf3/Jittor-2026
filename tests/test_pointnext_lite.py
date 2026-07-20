from pathlib import Path
import os
import unittest

import jittor as jt
import numpy as np
from omegaconf import OmegaConf

from src.model.feature import PointNeXtLiteHierarchy
from src.model.straightpcf import StraightPCFModule, VelocityNet


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
        self.assertEqual(cvm.hierarchy_hidden_dim, 64)
        self.assertEqual(cvm.hierarchy_stride, 4)
        self.assertEqual(cvm.hierarchy_k, 16)
        self.assertNotIn("encoder_type", control)

        for name in (
            "train_spcfgfnpnx001_cvm.yaml",
            "train_spcfgfnpnx001.yaml",
            "predict_spcfgfnpnx001a105_local2.yaml",
            "predict_spcfgfnpnx001a105_local2_pass2.yaml",
            "predict_spcfgfnpnx001a105_submit.yaml",
            "predict_spcfgfnpnx001a105_submit_pass2.yaml",
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

    def test_submission_pipeline_contract(self):
        pipeline = (
            ROOT / "scripts" / "run_pnx001_ensemble_candidate_pipeline.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--seed 123", pipeline)
        self.assertIn("--weights 0.70,0.30", pipeline)
        self.assertIn("--expected-count 200", pipeline)
        self.assertIn("--score-feature cv", pipeline)
        self.assertIn("--gamma 0.50", pipeline)
        self.assertNotIn("submit_educoder.py --yes", pipeline)

    @unittest.skipUnless(
        os.environ.get("PNX_CUDA_SMOKE") == "1",
        "set PNX_CUDA_SMOKE=1 for the full per-rank CUDA training smoke",
    )
    def test_full_batch_cuda_forward_backward(self):
        jt.flags.use_cuda = 1
        # Dataset.batch_size is global under Jittor MPI. The formal 32/4 setup
        # therefore executes eight samples per rank/GPU.
        batch_size = int(os.environ.get("PNX_SMOKE_BATCH", "8"))
        point_count = int(os.environ.get("PNX_SMOKE_POINTS", "1000"))
        model_name = os.environ.get("PNX_SMOKE_MODEL", "spcfgfnpnx001_cvm")
        model_config = OmegaConf.load(
            ROOT / "configs" / "model" / f"{model_name}.yaml"
        )
        transform_config = OmegaConf.load(
            ROOT / "configs" / "transform" / "spcf_n_rot001.yaml"
        )
        model = StraightPCFModule(model_config, transform_config)
        checkpoint = (
            ROOT / "experiments" / "_bak_official_75.01" / "cvm_checkpoint_best.pkl"
        )
        self.assertTrue(checkpoint.is_file())
        model.init_from_stage(str(checkpoint))
        model.train()

        rng = np.random.RandomState(29)
        clean = rng.randn(batch_size, point_count, 3).astype(np.float32) * 0.05
        noisy = clean + rng.laplace(
            0.0,
            0.011,
            size=clean.shape,
        ).astype(np.float32)
        time_step = rng.uniform(1e-8, 1.0, size=batch_size).astype(np.float32)
        blend = time_step.reshape(batch_size, 1, 1)
        seed = blend * clean[:, :1, :] + (1.0 - blend) * noisy[:, :1, :]
        batch = {
            "pcl_clean": jt.array(clean),
            "pcl_noisy_L2": jt.array(noisy),
            "seed_points_t": jt.array(seed),
            "original_time_step": jt.array(time_step),
        }
        optimizer = jt.optim.Adam(model.parameters(), lr=3e-5)
        loss = model.training_step(batch)["loss"]
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()
        jt.sync_all()
        jt.display_memory_info()
        loss_value = float(loss.item())
        print(
            f"PNX CUDA smoke model={model_name} batch={batch_size} "
            f"points={point_count} "
            f"loss={loss_value:.6f}"
        )
        self.assertTrue(np.isfinite(loss_value))


if __name__ == "__main__":
    unittest.main()
