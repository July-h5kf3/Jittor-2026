from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SHA = "06f5d3d271555682c95aa3505518f47eeab2bd9c"


class AscendEnvironmentContractTests(unittest.TestCase):
    def test_activation_is_pinned(self):
        text = (ROOT / "scripts/ascend_env.sh").read_text(encoding="utf-8")
        self.assertIn("/usr/local/Ascend/cann-9.1.0-beta.1", text)
        self.assertIn(SHA, text)
        self.assertIn("/data/ldc/cache/jittor-track2-ascend", text)

    def test_remote_setup_is_offline(self):
        text = (ROOT / "scripts/setup_ascend_env.sh").read_text(encoding="utf-8")
        self.assertIn("--no-index", text)
        self.assertIn("--no-build-isolation", text)
        self.assertIn(f"jittor-{SHA}", text)
        self.assertNotIn("github.com", text)

    def test_sync_excludes_generated_state(self):
        text = (ROOT / "scripts/sync_ascend.sh").read_text(encoding="utf-8")
        for value in (".git", "__pycache__", "*.pyc", ".venv", "train_logs"):
            self.assertIn(value, text)
        self.assertIn("zhiyuan-huawei:/data/ldc/Track2-new/", text)


if __name__ == "__main__":
    unittest.main()
