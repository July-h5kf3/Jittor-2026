import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "calibrate_adaptive_two_pass",
    ROOT / "scripts" / "calibrate_adaptive_two_pass.py",
)
calibrate_adaptive_two_pass = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = calibrate_adaptive_two_pass
SPEC.loader.exec_module(calibrate_adaptive_two_pass)


class AdaptiveTwoPassTests(unittest.TestCase):
    def test_robust_scores_order_and_clip(self):
        scores = calibrate_adaptive_two_pass._robust_scores(
            {"low": 0.1, "mid": 1.0, "high": 100.0}
        )
        self.assertLess(scores["low"], scores["mid"])
        self.assertLess(scores["mid"], scores["high"])
        self.assertGreaterEqual(scores["low"], -1.0)
        self.assertLessEqual(scores["high"], 1.0)

    def test_cv_statistic_is_scale_invariant(self):
        norms = np.asarray([1.0, 2.0, 5.0], dtype=np.float64)
        first = calibrate_adaptive_two_pass._correction_statistic(norms, "cv")
        second = calibrate_adaptive_two_pass._correction_statistic(
            7.0 * norms,
            "cv",
        )
        self.assertAlmostEqual(first, second)

    def test_mean_statistic_preserves_existing_behavior(self):
        norms = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
        value = calibrate_adaptive_two_pass._correction_statistic(norms, "mean")
        self.assertAlmostEqual(value, 2.0)

    def test_raw_alpha_gate_metadata_is_optional(self):
        clipped = {"cloud": 0.97}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            legacy = root / "legacy.tsv"
            legacy.write_text("key\talpha\ncloud\t0.97\n", encoding="utf-8")
            values, uses_raw = calibrate_adaptive_two_pass._load_gate_alphas(
                legacy, clipped
            )
            self.assertFalse(uses_raw)
            self.assertEqual(values, clipped)

            modern = root / "modern.tsv"
            modern.write_text(
                "key\talpha\traw_alpha\ncloud\t0.97\t0.9698\n",
                encoding="utf-8",
            )
            values, uses_raw = calibrate_adaptive_two_pass._load_gate_alphas(
                modern, clipped
            )
            self.assertTrue(uses_raw)
            self.assertAlmostEqual(values["cloud"], 0.9698)

    def test_raw_alpha_gate_keeps_marginal_clip_on_regular_schedule(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = root / "first"
            adaptive = root / "adaptive"
            second = root / "second"
            output = root / "output"
            for key in ("low", "marginal"):
                for directory, value in (
                    (first, 0.0),
                    (adaptive, 0.0),
                    (second, 0.1),
                ):
                    target = directory / key
                    target.mkdir(parents=True)
                    np.save(
                        target / "denoised.npy",
                        np.full((4, 3), value, dtype=np.float32),
                    )
            alpha_manifest = root / "alpha.tsv"
            alpha_manifest.write_text(
                "key\talpha\traw_alpha\n"
                "low\t0.97\t0.95\n"
                "marginal\t0.97\t0.9698\n",
                encoding="utf-8",
            )
            calibrate_adaptive_two_pass.fuse_tree(
                first,
                adaptive,
                second,
                output,
                alpha_manifest,
                2,
                0.97,
                0.96,
                -0.30,
                0.45,
                0.50,
                "cv",
                0.36,
                0.54,
                root / "manifest.tsv",
            )
            low = np.load(output / "low" / "denoised.npy")
            marginal = np.load(output / "marginal" / "denoised.npy")
            np.testing.assert_allclose(low, -0.03, atol=1e-6)
            np.testing.assert_allclose(marginal, 0.045, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
