import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "summarize_comparison", ROOT / "scripts" / "summarize_comparison.py"
)
summarize_comparison = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = summarize_comparison
SPEC.loader.exec_module(summarize_comparison)


class SummarizeComparisonTests(unittest.TestCase):
    def _rows(self):
        return {
            "control": {
                "shapenet/a/one": {
                    "cd_score": 10.0,
                    "p2s_score": 20.0,
                    "total_score": 15.0,
                },
                "shapenet/b/two": {
                    "cd_score": 30.0,
                    "p2s_score": 40.0,
                    "total_score": 35.0,
                },
            },
            "candidate": {
                "shapenet/a/one": {
                    "cd_score": 11.0,
                    "p2s_score": 22.0,
                    "total_score": 16.5,
                },
                "shapenet/b/two": {
                    "cd_score": 31.0,
                    "p2s_score": 42.0,
                    "total_score": 36.5,
                },
            },
        }

    def test_paired_category_and_loco_summary(self):
        summary = summarize_comparison.summarize(
            self._rows(), "control", ["candidate"], 200, 7
        )
        candidate = summary["candidates"]["candidate"]
        self.assertEqual(candidate["wins"], 2)
        self.assertAlmostEqual(
            candidate["metrics"]["total_score"]["delta"], 1.5
        )
        self.assertEqual(candidate["categories"]["a"]["count"], 1)
        self.assertAlmostEqual(
            candidate["leave_one_category_out"]["a"]["metrics"]
            ["total_score"]["delta"],
            1.5,
        )
        self.assertEqual(candidate["loco_total_delta_range"], [1.5, 1.5])

    def test_load_rows_rejects_duplicate_keys(self):
        content = (
            "label\tkey\tcd_score\tp2s_score\ttotal_score\n"
            "a\tshapenet/x/one\t1\t2\t1.5\n"
            "a\tshapenet/x/one\t1\t2\t1.5\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.tsv"
            path.write_text(content, encoding="utf-8")
            with self.assertRaises(summarize_comparison.SummaryError):
                summarize_comparison.load_rows(path)


if __name__ == "__main__":
    unittest.main()
