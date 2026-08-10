from pathlib import Path
import subprocess
import tempfile
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
        self.assertIn('if [ ! -r "$ASCEND_TOOLKIT_HOME/set_env.sh" ]; then', text)
        self.assertIn("return 1 2>/dev/null || exit 1", text)

    def test_activation_stops_on_cann_source_failure_without_changing_options(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            cann_root = temp / "cann"
            cann_root.mkdir()
            (cann_root / "set_env.sh").write_text("return 23\n", encoding="utf-8")
            activation = self._activation_copy(temp, cann_root)
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    'set -o nounset; before=$(set +o); '
                    'if source "$1"; then status=0; else status=$?; fi; '
                    'after=$(set +o); printf "%s\\n" "$status"; '
                    '[ "$before" = "$after" ] && [ -z "${JITTOR_HOME+x}" ]',
                    "bash",
                    str(activation),
                ],
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "1\n")
            self.assertIn("error: failed to source CANN environment script", result.stderr)

    def test_activation_succeeds_and_exports_expected_variables(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            cann_root = temp / "cann"
            cann_root.mkdir()
            (cann_root / "set_env.sh").write_text(
                "export FAKE_CANN_READY=1\n", encoding="utf-8"
            )
            activation = self._activation_copy(temp, cann_root)
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    'set -o nounset; source "$1"; '
                    'printf "%s|%s|%s\\n" "$JITTOR_HOME" "$NKAI_JITTOR_COMMIT" "$FAKE_CANN_READY"',
                    "bash",
                    str(activation),
                ],
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                result.stdout,
                f"/data/ldc/cache/jittor-track2-ascend|{SHA}|1\n",
            )

    def test_remote_setup_is_offline(self):
        text = (ROOT / "scripts/setup_ascend_env.sh").read_text(encoding="utf-8")
        self.assertEqual(text.count("--no-index"), 2)
        self.assertIn("--no-build-isolation", text)
        self.assertIn("--no-deps -e", text)
        self.assertIn(f"jittor-{SHA}", text)
        self.assertNotIn("github.com", text)

    def test_remote_setup_checks_jittor_revision_and_runtime(self):
        text = (ROOT / "scripts/setup_ascend_env.sh").read_text(encoding="utf-8")
        self.assertIn(
            'git -c safe.directory="$JITTOR_ROOT" -C "$JITTOR_ROOT" rev-parse HEAD',
            text,
        )
        self.assertIn('"$actual_jittor_sha" != "$JITTOR_SHA"', text)
        self.assertIn("python3.11 -m venv", text)
        self.assertIn("sys.version_info[:2] != (3, 11)", text)
        self.assertIn('platform.machine().lower() != "aarch64"', text)
        self.assertIn(
            'git -c safe.directory="$JITTOR_ROOT" -C "$JITTOR_ROOT" status --porcelain',
            text,
        )
        self.assertNotIn('test -f "$PROJECT_ROOT/requirements-ascend.txt"', text)

    def test_remote_setup_scopes_git_safe_directory_to_jittor_checkout(self):
        text = (ROOT / "scripts/setup_ascend_env.sh").read_text(encoding="utf-8")
        self.assertEqual(
            text.count('git -c safe.directory="$JITTOR_ROOT" -C "$JITTOR_ROOT"'),
            2,
        )
        self.assertNotIn("git config --global", text)

    def test_jittor_source_validation_accepts_only_clean_matching_revision(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            jittor_root = temp / "jittor"
            jittor_root.mkdir()
            (jittor_root / "setup.py").write_text("# fixture\n", encoding="utf-8")
            self._git(jittor_root, "init")
            self._git(jittor_root, "add", "setup.py")
            self._git(
                jittor_root,
                "-c",
                "user.email=test@example.invalid",
                "-c",
                "user.name=Test",
                "commit",
                "-m",
                "fixture",
            )
            matching_sha = self._git(jittor_root, "rev-parse", "HEAD").stdout.strip()

            with self.subTest("clean matching revision"):
                result = self._run_setup_validation(temp, jittor_root, matching_sha)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, "validation-passed\n")

            with self.subTest("wrong revision"):
                result = self._run_setup_validation(temp, jittor_root, "0" * 40)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Jittor revision is", result.stderr)

            with self.subTest("tracked changes"):
                (jittor_root / "setup.py").write_text("# modified\n", encoding="utf-8")
                result = self._run_setup_validation(temp, jittor_root, matching_sha)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("working tree is not clean", result.stderr)
                self._git(jittor_root, "checkout", "--", "setup.py")

            with self.subTest("untracked changes"):
                (jittor_root / "local.patch").write_text("patch\n", encoding="utf-8")
                result = self._run_setup_validation(temp, jittor_root, matching_sha)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("working tree is not clean", result.stderr)

    def _activation_copy(self, temp: Path, cann_root: Path) -> Path:
        text = (ROOT / "scripts/ascend_env.sh").read_text(encoding="utf-8")
        text = text.replace("/usr/local/Ascend/cann-9.1.0-beta.1", str(cann_root))
        activation = temp / "ascend_env.sh"
        activation.write_text(text, encoding="utf-8")
        return activation

    def _run_setup_validation(
        self, temp: Path, jittor_root: Path, expected_sha: str
    ) -> subprocess.CompletedProcess[str]:
        project_root = temp / "project"
        wheel_root = temp / "wheels"
        env_root = temp / "env"
        project_root.mkdir(exist_ok=True)
        wheel_root.mkdir(exist_ok=True)
        (project_root / "requirements-ascend.txt").write_text("", encoding="utf-8")
        text = (ROOT / "scripts/setup_ascend_env.sh").read_text(encoding="utf-8")
        text = text.replace("/data/ldc/Track2-new", str(project_root))
        text = text.replace("/data/ldc/envs/track2-ascend", str(env_root))
        text = text.replace("/data/ldc/packages/track2-ascend", str(wheel_root))
        text = text.replace(f"JITTOR_SHA={SHA}", f"JITTOR_SHA={expected_sha}")
        text = text.replace(f"/data/ldc/vendor/jittor-{SHA}", str(jittor_root))
        text = text.replace(
            "mkdir -p /data/ldc/envs /data/ldc/cache/jittor-track2-ascend",
            "printf 'validation-passed\\n'\nexit 0",
        )
        script = temp / "setup_validation.sh"
        script.write_text(text, encoding="utf-8")
        return subprocess.run(["bash", str(script)], text=True, capture_output=True)

    def _git(self, cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(cwd), *args],
            check=True,
            text=True,
            capture_output=True,
        )

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
