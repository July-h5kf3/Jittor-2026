from pathlib import Path
import unittest

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]


class BestConfigTests(unittest.TestCase):
    def test_best_cvm_contract(self):
        cfg = OmegaConf.load(ROOT / "configs/model/spcfgfncvm002_cvm.yaml")
        self.assertEqual(cfg.stage, "cvm")
        self.assertEqual(cfg.num_modules, 4)
        self.assertEqual(cfg.decoder_type, "graph")
        self.assertTrue(cfg.film)
        self.assertEqual(cfg.cvm_dir_target, "stage_velocity")
        self.assertTrue(cfg.cvm_deep_sup)

    def test_best_spcf_contract(self):
        cfg = OmegaConf.load(ROOT / "configs/model/spcfgfncvm002_spcf.yaml")
        self.assertEqual(cfg.stage, "spcf")
        self.assertTrue(cfg.distance_multiscale)
        self.assertEqual(cfg.predict_alpha, 1.0)
        self.assertEqual(cfg.predict_passes, 1)
        self.assertEqual(cfg.predict_tta, 0)
        self.assertFalse(cfg.predict_fusion)

    def test_task_component_files_exist(self):
        for name in (
            "train_spcfgfncvm002_cvm.yaml",
            "train_spcfgfncvm002.yaml",
            "predict_spcfgfncvm002.yaml",
            "predict_spcfgfncvm002_local2.yaml",
            "train_spcfgfnldc.yaml",
            "predict_spcfgfnldc.yaml",
            "predict_spcfgfnldc_local2.yaml",
        ):
            task = OmegaConf.load(ROOT / "configs/task" / name)
            for kind in ("data", "transform", "system", "model"):
                component = task.components[kind]
                self.assertTrue(
                    (ROOT / "configs" / kind / f"{component}.yaml").is_file(),
                    f"missing {kind}/{component}.yaml referenced by {name}",
                )

    def test_ldc_is_zero_risk_finetune_of_best_spcf(self):
        model = OmegaConf.load(ROOT / "configs/model/spcfgfnldc_spcf.yaml")
        task = OmegaConf.load(ROOT / "configs/task/train_spcfgfnldc.yaml")
        self.assertEqual(model.stage, "spcf")
        self.assertEqual(model.cvm_dir_target, "stage_velocity")
        self.assertTrue(model.cvm_deep_sup)
        self.assertEqual(model.cvm_condition, "time_stage")
        self.assertTrue(model.spcf_train_condition_adapter)
        self.assertEqual(model.spcf_train_unroll_its, model.tot_its)
        self.assertEqual(
            task.load_ckpt,
            "experiments/spcfgfncvm002_spcf/checkpoint_best.pkl",
        )
        self.assertLessEqual(task.optimizer.lr, 0.00001)
        self.assertNotIn("initial_best_metric", task.trainer)


if __name__ == "__main__":
    unittest.main()
