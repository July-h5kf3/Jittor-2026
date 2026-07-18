import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "calibrate_predictions", ROOT / "scripts" / "calibrate_predictions.py"
)
calibrate_predictions = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(calibrate_predictions)


class AdaptiveAlphaTests(unittest.TestCase):
    def test_mean_var_profile_and_cloud_calibration(self):
        noisy = np.zeros((4, 3), dtype=np.float64)
        prediction = np.asarray(
            [[0.01, 0, 0], [0.02, 0, 0], [0.03, 0, 0], [0.04, 0, 0]],
            dtype=np.float64,
        )
        features = calibrate_predictions.extract_features(
            noisy, prediction, "category/model"
        )
        alpha = calibrate_predictions.estimate_alpha(features, "mean-var")
        raw_alpha = calibrate_predictions.estimate_raw_alpha(
            features, "mean-var"
        )
        self.assertGreaterEqual(alpha, calibrate_predictions.ALPHA_MIN)
        self.assertLessEqual(alpha, calibrate_predictions.ALPHA_MAX)
        self.assertAlmostEqual(
            alpha,
            float(
                np.clip(
                    raw_alpha,
                    calibrate_predictions.ALPHA_MIN,
                    calibrate_predictions.ALPHA_MAX,
                )
            ),
        )
        output = calibrate_predictions.calibrate_cloud(noisy, prediction, alpha)
        np.testing.assert_allclose(
            output, prediction * (alpha / calibrate_predictions.BASE_ALPHA)
        )
        self.assertEqual(output.dtype, np.float32)

    def test_tree_calibration_writes_only_matched_predictions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pred = root / "pred"
            noisy = root / "noisy"
            out = root / "out"
            key = Path("category") / "model"
            (pred / key).mkdir(parents=True)
            (noisy / key).mkdir(parents=True)
            base = np.zeros((8, 3), dtype=np.float32)
            estimate = np.full((8, 3), 0.01, dtype=np.float32)
            np.save(pred / key / "denoised.npy", estimate)
            np.save(noisy / key / "noisy.npy", base)
            manifest = root / "manifest.tsv"
            calibrate_predictions.calibrate_tree(
                pred, noisy, out, "mean-var", 1.05, 1, manifest
            )
            result = np.load(out / key / "denoised.npy")
            self.assertEqual(result.shape, (8, 3))
            self.assertTrue(manifest.is_file())
            manifest_text = manifest.read_text(encoding="utf-8")
            self.assertIn("raw_alpha", manifest_text.splitlines()[0])
            self.assertIn("category/model", manifest_text)


if __name__ == "__main__":
    unittest.main()
