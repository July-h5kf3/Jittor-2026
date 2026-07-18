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
    def test_default_archive_is_current_candidate(self):
        args = submit_educoder.parse_args([])
        self.assertEqual(
            args.zip_path,
            "submission_results/result_cvm002a105_adaptive_meanvar_a110.zip",
        )

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

    def test_clock_offset_bypasses_cached_homepage(self):
        client = object.__new__(submit_educoder.EducoderClient)
        response = mock.Mock()
        response.headers = {"Date": "Tue, 14 Jul 2026 02:46:49 GMT"}
        client.http = mock.Mock()
        client.http.get.return_value = response

        with mock.patch.object(submit_educoder.time, "time", return_value=0):
            offset = client._get_clock_offset()

        self.assertGreater(offset, 0)
        _, kwargs = client.http.get.call_args
        self.assertIn("_clock", kwargs["params"])
        self.assertEqual(kwargs["headers"], {"Cache-Control": "no-cache"})

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

    def test_status_only_skips_archive_validation_and_upload_token(self):
        class Client:
            def __init__(self, cookie, competition):
                pass

            def user_info(self):
                return {"login": "tester"}

            def teams(self):
                return [{"id": 205141, "name": "team"}]

            def results(self, stage_id):
                return []

        with mock.patch.dict(
            os.environ, {"EDUCODER_COOKIE": "_educoder_session=x"}
        ), mock.patch.object(
            submit_educoder, "EducoderClient", Client
        ), mock.patch.object(
            submit_educoder, "validate_archive"
        ) as validate:
            result = submit_educoder.main(["missing.zip", "--status-only"])

        self.assertEqual(result, 0)
        validate.assert_not_called()

    def test_uncertain_callback_uses_created_result(self):
        archive = {
            "path": "result.zip",
            "file_name": "result.zip",
            "size": 123,
            "sha256": "0" * 64,
            "count": 1,
            "points": 2,
        }
        created = {
            "id": 7,
            "file_name": "result.zip",
            "status": 0,
            "data_ranking": None,
        }

        class Client:
            def __init__(self, cookie, competition):
                pass

            def user_info(self):
                return {"login": "tester"}

            def teams(self):
                return [{"id": 205141, "name": "team"}]

            def results(self, stage_id):
                return []

            def upload_token(self):
                return {
                    "bucket": "bucket",
                    "end_point": "endpoint",
                    "bucket_host": "host",
                }

        stdout = io.StringIO()
        with mock.patch.dict(os.environ, {"EDUCODER_COOKIE": "session=x"}), mock.patch.object(
            submit_educoder, "validate_archive", return_value=archive
        ), mock.patch.object(
            submit_educoder, "EducoderClient", Client
        ), mock.patch.object(
            submit_educoder,
            "upload_archive",
            side_effect=submit_educoder.CallbackUncertain("non-JSON"),
        ), mock.patch.object(
            submit_educoder, "_find_new_result", return_value=created
        ), redirect_stdout(stdout):
            result = submit_educoder.main(["result.zip", "--yes"])

        self.assertEqual(result, 0)
        self.assertIn("callback response uncertain", stdout.getvalue())
        self.assertIn("submission created: id=7", stdout.getvalue())

    def test_wait_for_result_reports_completed_score(self):
        class Client:
            calls = 0

            def results(self, stage_id):
                self.calls += 1
                status = 0 if self.calls == 1 else 2
                score = None if status == 0 else 76.5
                return [
                    {
                        "id": 7,
                        "status": status,
                        "data_ranking": score,
                        "updated_at": "now",
                    }
                ]

        with mock.patch.object(submit_educoder.time, "sleep"):
            result = submit_educoder.wait_for_result(Client(), 588, 7, 10, 1)

        self.assertEqual(result["data_ranking"], 76.5)

    def test_wait_for_result_retries_transient_http_error(self):
        class Client:
            calls = 0

            def results(self, stage_id):
                self.calls += 1
                if self.calls == 1:
                    raise submit_educoder.requests.HTTPError("502")
                return [
                    {
                        "id": 7,
                        "status": 2,
                        "data_ranking": 76.5,
                        "updated_at": "now",
                    }
                ]

        with mock.patch.object(submit_educoder.time, "sleep"):
            result = submit_educoder.wait_for_result(Client(), 588, 7, 10, 1)

        self.assertEqual(result["status"], 2)

    def test_find_new_result_retries_transient_http_error(self):
        class Client:
            calls = 0

            def results(self, stage_id):
                self.calls += 1
                if self.calls == 1:
                    raise submit_educoder.requests.HTTPError("502")
                return [{"id": 7, "file_name": "result.zip"}]

        with mock.patch.object(submit_educoder.time, "sleep"):
            result = submit_educoder._find_new_result(
                Client(), 588, "result.zip", set(), attempts=2
            )

        self.assertEqual(result["id"], 7)


if __name__ == "__main__":
    unittest.main()
