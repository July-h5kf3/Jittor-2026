import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "calibrate_two_pass", ROOT / "scripts" / "calibrate_two_pass.py"
)
calibrate_two_pass = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(calibrate_two_pass)


class TwoPassTests(unittest.TestCase):
    def test_aligns_nested_writer_prefix(self):
        aligned = calibrate_two_pass.align_predictions(
            "second",
            {"localtest2/shapenet/a"},
            {
                "results_first/localtest2/shapenet/a": Path("second.npy")
            },
        )
        self.assertEqual(
            aligned["localtest2/shapenet/a"], Path("second.npy")
        )

    def test_lower_bound_uses_negative_beta(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            adaptive = root / "adaptive"
            second = root / "second"
            for base, value in ((first, 1.0), (adaptive, 2.0), (second, 3.0)):
                target = base / "cat" / "model" / "denoised.npy"
                target.parent.mkdir(parents=True)
                np.save(target, np.full((4, 3), value, dtype=np.float32))
            alpha = root / "alpha.tsv"
            alpha.write_text("key\talpha\ncat/model\t0.97000000\n", encoding="utf-8")
            output = root / "output"
            calibrate_two_pass.fuse_tree(
                first,
                adaptive,
                second,
                output,
                alpha,
                1,
                0.97,
                -0.30,
                0.45,
                root / "manifest.tsv",
            )
            points = np.load(output / "cat" / "model" / "denoised.npy")
            np.testing.assert_allclose(points, 1.4)


if __name__ == "__main__":
    unittest.main()
