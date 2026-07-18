import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "calibrate_uncertainty", ROOT / "scripts" / "calibrate_uncertainty.py"
)
calibrate_uncertainty = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = calibrate_uncertainty
SPEC.loader.exec_module(calibrate_uncertainty)


class UncertaintyCalibrationTests(unittest.TestCase):
    def test_zero_gamma_is_meanvar_calibrated_ensemble(self):
        noisy = np.zeros((4, 3), dtype=np.float64)
        first = np.full((4, 3), 0.10, dtype=np.float64)
        second = np.full((4, 3), 0.20, dtype=np.float64)

        output, alpha, _, factors = calibrate_uncertainty.calibrate_cloud(
            noisy, [first, second], gamma=0.0
        )

        np.testing.assert_allclose(factors, 1.0)
        expected = (alpha / calibrate_uncertainty.BASE_ALPHA) * 0.15
        np.testing.assert_allclose(output, expected, rtol=1e-6)

    def test_positive_gamma_reduces_high_disagreement_step(self):
        noisy = np.zeros((4, 3), dtype=np.float64)
        first = np.full((4, 3), 0.10, dtype=np.float64)
        second = first.copy()
        second[-1] = 0.30

        _, _, disagreement, factors = calibrate_uncertainty.calibrate_cloud(
            noisy, [first, second], gamma=0.2
        )

        self.assertGreater(disagreement[-1], disagreement[0])
        self.assertLess(factors[-1], factors[0])


if __name__ == "__main__":
    unittest.main()
