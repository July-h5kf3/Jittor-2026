import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class SourcePolicyTests(unittest.TestCase):
    def test_all_python_has_no_prohibited_runtime(self):
        prohibited = (
            "torch",
            "triton",
            "mamba_ssm",
            "causal_conv1d",
            "scipy",
            "trimesh",
            "omegaconf",
            "tqdm",
        )
        pattern = re.compile(
            r"^\s*(?:from\s+(?:" + "|".join(prohibited) + r")\b"
            r"|import\s+(?:" + "|".join(prohibited) + r")\b)",
            re.MULTILINE,
        )
        paths = ROOT.rglob("*.py")
        for path in paths:
            self.assertFalse(pattern.search(path.read_text(encoding="utf-8")), str(path))

    def test_no_embedded_artifacts(self):
        forbidden = {".pth", ".pt", ".ckpt", ".pkl", ".npy", ".npz"}
        bad = [path for path in ROOT.rglob("*") if path.is_file() and path.suffix.lower() in forbidden]
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
