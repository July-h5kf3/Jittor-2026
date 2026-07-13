import importlib.util
import io
import os
import sys
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

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

    def test_dry_run_allows_existing_file_name(self):
        archive = {
            "path": "result.zip",
            "file_name": "result.zip",
            "size": 123,
            "sha256": "0" * 64,
            "count": 1,
            "points": 2,
        }

        class Client:
            def __init__(self, cookie, competition):
                pass

            def user_info(self):
                return {"login": "tester"}

            def teams(self):
                return [{"id": 205141, "name": "team"}]

            def results(self, stage_id):
                return [{"id": 1, "file_name": "result.zip", "status": 2}]

            def upload_token(self):
                return {
                    "bucket": "bucket",
                    "end_point": "endpoint",
                    "bucket_host": "host",
                }

        stdout = io.StringIO()
        with mock.patch.dict(os.environ, {"EDUCODER_COOKIE": "session=x"}), mock.patch.object(
            submit_educoder, "validate_archive", return_value=archive
        ), mock.patch.object(submit_educoder, "EducoderClient", Client), redirect_stdout(stdout):
            result = submit_educoder.main(["result.zip", "--dry-run"])

        self.assertEqual(result, 0)
        self.assertIn("dry-run warning", stdout.getvalue())
        self.assertIn("dry-run complete", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
