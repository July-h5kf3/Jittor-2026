import pathlib
import re
import tempfile
import unittest

from tools.audit_source_archive import audit_tree


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

    def test_source_audit_ignores_git_file_or_directory(self):
        for git_entry_type in ("file", "directory"):
            with self.subTest(git_entry_type=git_entry_type):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    root = pathlib.Path(temporary_directory)
                    (root / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
                    expected = audit_tree(root)

                    git_entry = root / ".git"
                    if git_entry_type == "file":
                        git_entry.write_text(
                            "gitdir: /tmp/example-worktree\n", encoding="utf-8"
                        )
                    else:
                        git_entry.mkdir()
                        (git_entry / "config").write_text(
                            "[core]\n\trepositoryformatversion = 0\n",
                            encoding="utf-8",
                        )

                    self.assertEqual(audit_tree(root), expected)


if __name__ == "__main__":
    unittest.main()
