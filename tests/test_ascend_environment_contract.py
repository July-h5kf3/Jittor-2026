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

    def test_activation_is_safe_to_source(self):
        text = (ROOT / "scripts/ascend_env.sh").read_text(encoding="utf-8")
        self.assertNotIn("set -euo pipefail", text)
        self.assertNotIn("set -u", text)
        self.assertIn('if [ ! -f "$ASCEND_TOOLKIT_HOME/set_env.sh" ]; then', text)
        self.assertIn("return 1 2>/dev/null || exit 1", text)

    def test_remote_setup_is_offline(self):
        text = (ROOT / "scripts/setup_ascend_env.sh").read_text(encoding="utf-8")
        self.assertEqual(text.count("--no-index"), 2)
        self.assertIn("--no-build-isolation", text)
        self.assertIn("--no-deps -e", text)
        self.assertIn(f"jittor-{SHA}", text)
        self.assertNotIn("github.com", text)

    def test_remote_setup_checks_jittor_revision_and_runtime(self):
        text = (ROOT / "scripts/setup_ascend_env.sh").read_text(encoding="utf-8")
        self.assertIn('git -C "$JITTOR_ROOT" rev-parse HEAD', text)
        self.assertIn('"$actual_jittor_sha" != "$JITTOR_SHA"', text)
        self.assertIn("python3.11 -m venv", text)
        self.assertIn("sys.version_info[:2] != (3, 11)", text)
        self.assertIn('platform.machine().lower() != "aarch64"', text)

    def test_sync_excludes_generated_state(self):
        text = (ROOT / "scripts/sync_ascend.sh").read_text(encoding="utf-8")
        for value in (
            ".git",
            "__pycache__",
            "*.pyc",
            ".venv",
            "train_logs",
            "checkpoints",
            "predictions",
        ):
            self.assertIn(value, text)
        self.assertIn("zhiyuan-huawei:/data/ldc/Track2-new/", text)
        self.assertIn('${BASH_SOURCE[0]}', text)


if __name__ == "__main__":
    unittest.main()
