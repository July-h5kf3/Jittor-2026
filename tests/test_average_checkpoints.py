import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "average_checkpoints", ROOT / "scripts" / "average_checkpoints.py"
)
average_checkpoints = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = average_checkpoints
SPEC.loader.exec_module(average_checkpoints)


class AverageCheckpointsTests(unittest.TestCase):
    def test_weighted_average_preserves_float_dtype(self):
        first = {"weight": np.asarray([1.0, 3.0], dtype=np.float32)}
        second = {"weight": np.asarray([5.0, 7.0], dtype=np.float32)}
        result = average_checkpoints.average_state_dicts(
            [first, second], [3.0, 1.0]
        )
        self.assertEqual(result["weight"].dtype, np.float32)
        np.testing.assert_allclose(result["weight"], [2.0, 4.0])

    def test_rejects_mismatched_keys(self):
        with self.assertRaises(average_checkpoints.CheckpointAverageError):
            average_checkpoints.average_state_dicts(
                [{"a": np.ones(1)}, {"b": np.ones(1)}], [1.0, 1.0]
            )


if __name__ == "__main__":
    unittest.main()
