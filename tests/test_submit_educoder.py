import importlib.util
import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "submit_educoder", ROOT / "scripts" / "submit_educoder.py"
)
submit_educoder = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = submit_educoder
SPEC.loader.exec_module(submit_educoder)


class SubmitEducoderTests(unittest.TestCase):
    def test_cookie_parser(self):
        cookie = "autologin_trustie=abc; _educoder_session=session-value"
        self.assertEqual(
            submit_educoder._session_from_cookie(cookie), "session-value"
        )

    def test_callback_query_matches_browser_encoding(self):
        query = submit_educoder._callback_query(
            {"name": "a b.zip", "stage_type": 588, "result_id": None}
        )
        self.assertEqual(query, "name=a%20b.zip&stage_type=588")

    def test_read_oss_result_supports_put_object_result(self):
        class Response:
            def read(self):
                return b'{"status":0}'

        class Result:
            resp = Response()

        self.assertEqual(
            submit_educoder._read_oss_result(Result()), b'{"status":0}'
        )

    def test_validate_archive(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "result.zip"
            buffer = io.BytesIO()
            np.save(buffer, np.zeros((2, 3), dtype=np.float32))
            buffer.seek(0)
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(
                    "shapenet/000/abc/denoised.npy", buffer.read()
                )
            result = submit_educoder.validate_archive(path, 1, 2, 1.0)
            self.assertEqual(result["count"], 1)
            self.assertEqual(len(result["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
