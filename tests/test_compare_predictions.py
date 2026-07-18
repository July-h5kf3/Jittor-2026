import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "compare_predictions", ROOT / "scripts" / "compare_predictions.py"
)
compare_predictions = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = compare_predictions
SPEC.loader.exec_module(compare_predictions)


class ComparePredictionsTests(unittest.TestCase):
    def test_prediction_argument(self):
        label, path = compare_predictions._parse_prediction("ema=results/ema")
        self.assertEqual(label, "ema")
        self.assertEqual(path.name, "ema")

    def test_bootstrap_constant_difference(self):
        low, high = compare_predictions._bootstrap_ci(
            np.full(12, 0.25), 500, 7
        )
        self.assertAlmostEqual(low, 0.25)
        self.assertAlmostEqual(high, 0.25)

    def test_aligns_writer_input_prefix(self):
        mapping = {
            "localtest2/shapenet/a/denoised": Path("prediction.npy")
        }
        aligned = compare_predictions._align_predictions(
            "candidate", mapping, ["shapenet/a/denoised"]
        )
        self.assertEqual(
            aligned["shapenet/a/denoised"], Path("prediction.npy")
        )


if __name__ == "__main__":
    unittest.main()
