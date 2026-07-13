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
            "predict_spcfgfncvm002a105.yaml",
            "predict_spcfgfncvm002a105_local2.yaml",
        ):
            task = OmegaConf.load(ROOT / "configs/task" / name)
            for kind in ("data", "transform", "system", "model"):
                component = task.components[kind]
                self.assertTrue(
                    (ROOT / "configs" / kind / f"{component}.yaml").is_file(),
                    f"missing {kind}/{component}.yaml referenced by {name}",
                )

    def test_alpha105_submission_candidate_contract(self):
        model = OmegaConf.load(
            ROOT / "configs/model/spcfgfncvm002a105_spcf.yaml"
        )
        task = OmegaConf.load(
            ROOT / "configs/task/predict_spcfgfncvm002a105.yaml"
        )
        self.assertEqual(model.predict_alpha, 1.05)
        self.assertEqual(model.predict_passes, 1)
        self.assertEqual(model.predict_tta, 0)
        self.assertFalse(model.predict_fusion)
        self.assertEqual(
            task.load_ckpt,
            "experiments/spcfgfncvm002_spcf/checkpoint_best.pkl",
        )

    def test_ddp_launcher_uses_an_absolute_python_interpreter(self):
        launcher = (ROOT / "scripts/train_ddp.sh").read_text(encoding="utf-8")
        self.assertIn("PYTHON_BIN", launcher)
        self.assertIn('"$PYTHON_BIN" run.py', launcher)
        self.assertIn("JITTOR_CACHE_PER_RANK", launcher)
        self.assertIn('rank${rank}', launcher)
        self.assertIn("OMPI_MCA_orte_tmpdir_base", launcher)
        self.assertIn("DISABLE_MULTIPROCESSING", launcher)


if __name__ == "__main__":
    unittest.main()
