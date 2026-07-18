import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ensemble_predictions", ROOT / "scripts" / "ensemble_predictions.py"
)
ensemble_predictions = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = ensemble_predictions
SPEC.loader.exec_module(ensemble_predictions)


class EnsemblePredictionsTests(unittest.TestCase):
    def test_weighted_average_preserves_shape_and_float32(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first" / "shapenet" / "a"
            second = root / "second" / "shapenet" / "a"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            np.save(first / "denoised.npy", np.ones((3, 3), dtype=np.float32))
            np.save(
                second / "denoised.npy",
                np.full((3, 3), 3.0, dtype=np.float32),
            )
            output = root / "output"

            ensemble_predictions.ensemble_trees(
                [root / "first", root / "second"],
                [1.0, 3.0],
                output,
                "denoised.npy",
                1,
            )

            result = np.load(output / "shapenet" / "a" / "denoised.npy")
            self.assertEqual(result.dtype, np.float32)
            np.testing.assert_allclose(result, 2.5)

    def test_aligns_writer_input_prefix(self):
        aligned = ensemble_predictions._align_map(
            ["shapenet/a"],
            {"dataset_test_noisy/shapenet/a": Path("prediction.npy")},
        )
        self.assertEqual(aligned["shapenet/a"], Path("prediction.npy"))


if __name__ == "__main__":
    unittest.main()
