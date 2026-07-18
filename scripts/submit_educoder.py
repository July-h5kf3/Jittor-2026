#!/usr/bin/env python3
"""Validate and submit a Track2 archive to the Educoder competition.

Credentials are read from an environment variable and are never written to
disk.  Run with --dry-run first; an actual upload additionally requires --yes
or an interactive confirmation.
"""

from __future__ import annotations

import argparse
import base64
import email.utils
import hashlib
import io
import json
import os
import re
import sys
import time
import uuid
import zipfile
from http.cookies import SimpleCookie
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional
from urllib.parse import quote

import numpy as np
import requests


BASE_URL = "https://www.educoder.net"
DEFAULT_COMPETITION = "Jittor-7"
DEFAULT_CONTAINER_ID = 983
DEFAULT_STAGE_ID = 588
DEFAULT_TEAM_ID = 205141
DEFAULT_ARCHIVE = (
    "submission_results/result_cvm002a105_adaptive_meanvar_a110.zip"
)

# These are public constants embedded in Educoder's browser bundle.  They are
# used only to reproduce the request signature required by the public API.
ENCODED_AK = "WlRsa1pEVmlORE15TW1ZNVpqZGtPRE5rTURBNVpHVTVZbVpoTVRBd1l6TT0="
ENCODED_SK = "TW1VelpHRXdObUZsTWpaaVlUbG1OelpoTldRNFpETTFOVGMwTm1ZeVptVT0="
TOKEN_AES_KEY = b"bf3c199c2470cb477d907b1e0917c17b"
TOKEN_AES_IV = b"5183666c72eec9e4"

MEMBER_RE = re.compile(r"shapenet/[^/]+/[^/]+/denoised\.npy")


class SubmissionError(RuntimeError):
    pass


class CallbackUncertain(SubmissionError):
    """OSS upload completed, but its callback response could not be decoded."""


def _double_b64decode(value: str) -> str:
    return base64.b64decode(base64.b64decode(value)).decode("utf-8")


def _session_from_cookie(cookie_header: str) -> str:
    parsed = SimpleCookie()
    parsed.load(cookie_header)
    morsel = parsed.get("_educoder_session")
    if morsel is None or not morsel.value:
        raise SubmissionError("cookie does not contain _educoder_session")
    return morsel.value


def _js_encode_uri_component(value: Any) -> str:
    # Matches encodeURIComponent for the scalar values used by the callback.
    return quote(str(value), safe="-_.!~*'()")


def _callback_query(values: Mapping[str, Any]) -> str:
    fields: List[str] = []
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            fields.extend(
                f"{key}[]={_js_encode_uri_component(item)}" for item in value
            )
        elif not isinstance(value, (dict, set)):
            fields.append(f"{key}={_js_encode_uri_component(value)}")
    return "&".join(fields)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_archive(
    path: Path,
    expected_count: int,
    expected_points: int,
    max_size_gib: float,
) -> Dict[str, Any]:
    if not path.is_file():
        raise SubmissionError(f"archive does not exist: {path}")
    if path.stat().st_size > max_size_gib * 1024**3:
        raise SubmissionError(
            f"archive is larger than the {max_size_gib:g} GiB platform limit"
        )

    with zipfile.ZipFile(path, "r") as archive:
        infos = [item for item in archive.infolist() if not item.is_dir()]
        names = [item.filename for item in infos]
        if len(names) != expected_count:
            raise SubmissionError(
                f"expected {expected_count} ZIP members, found {len(names)}"
            )
        if len(set(names)) != len(names):
            raise SubmissionError("archive contains duplicate member names")
        bad_names = [name for name in names if MEMBER_RE.fullmatch(name) is None]
        if bad_names:
            raise SubmissionError(f"invalid member path: {bad_names[0]}")

        corrupt = archive.testzip()
        if corrupt is not None:
            raise SubmissionError(f"corrupt ZIP member: {corrupt}")

        for index, info in enumerate(infos, 1):
            with archive.open(info, "r") as stream:
                points = np.load(io.BytesIO(stream.read()), allow_pickle=False)
            if points.shape != (expected_points, 3):
                raise SubmissionError(
                    f"invalid shape {points.shape} in {info.filename}"
                )
            if points.dtype != np.float32:
                raise SubmissionError(
                    f"invalid dtype {points.dtype} in {info.filename}; expected float32"
                )
            if not np.isfinite(points).all():
                raise SubmissionError(f"non-finite value in {info.filename}")
            if index % 25 == 0 or index == len(infos):
                print(f"validated arrays: {index}/{len(infos)}", flush=True)

    digest = _sha256(path)
    sidecar = Path(f"{path}.sha256")
    if sidecar.exists():
        expected_digest = sidecar.read_text(encoding="utf-8").split()[0].lower()
        if expected_digest != digest:
            raise SubmissionError(
                f"SHA256 sidecar mismatch: expected {expected_digest}, got {digest}"
            )

    return {
        "path": str(path),
        "file_name": path.name,
        "size": path.stat().st_size,
        "sha256": digest,
        "count": expected_count,
        "points": expected_points,
    }


class EducoderClient:
    def __init__(self, cookie_header: str, competition: str) -> None:
        self.cookie_header = cookie_header
        self.session_token = _session_from_cookie(cookie_header)
        self.competition = competition
        self.ak = _double_b64decode(ENCODED_AK)
        self.sk = _double_b64decode(ENCODED_SK)
        self.http = requests.Session()
        self.http.trust_env = False
        self.http.headers.update({"Cookie": cookie_header})
        self.clock_offset = self._get_clock_offset()

    def _get_clock_offset(self) -> float:
        # The homepage is CDN cached and its Date header can lag real time by
        # tens of minutes.  A cache-busting query keeps signed API timestamps
        # aligned with the origin server.
        response = self.http.get(
            BASE_URL + "/",
            params={"_clock": str(uuid.uuid4())},
            headers={"Cache-Control": "no-cache"},
            timeout=30,
        )
        response.raise_for_status()
        date_header = response.headers.get("Date")
        if not date_header:
            raise SubmissionError("Educoder response did not contain a Date header")
        server_time = email.utils.parsedate_to_datetime(date_header).timestamp()
        return server_time - time.time()

    def _headers(self, method: str) -> Dict[str, str]:
        timestamp = str(int((time.time() + self.clock_offset) * 1000))
        plain = (
            f"method={method.upper()}&ak={self.ak}&sk={self.sk}&time={timestamp}"
        )
        signature = hashlib.md5(base64.b64encode(plain.encode("utf-8"))).hexdigest()
        return {
            "Accept": "application/json",
            "Content-Type": "application/json; charset=utf-8",
            "Origin": BASE_URL,
            "Referer": f"{BASE_URL}/competitions/{self.competition}",
            "Pc-Authorization": self.session_token,
            "X-EDU-Type": "pc",
            "X-EDU-Timestamp": timestamp,
            "X-EDU-Signature": signature,
            "X-Original-Protocol": "https:",
            "X-Original-Host": "www.educoder.net",
            "X-Original-Origin": BASE_URL,
            "X-Request-Id": str(uuid.uuid4()),
        }

    def get(self, path: str, params: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        response = self.http.get(
            BASE_URL + path,
            params=params,
            headers=self._headers("GET"),
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and payload.get("status") not in (None, 0):
            raise SubmissionError(
                f"Educoder API rejected {path}: {payload.get('message', payload)}"
            )
        return payload

    def user_info(self) -> Dict[str, Any]:
        return self.get("/api/users/get_user_info.json")

    def teams(self) -> List[Dict[str, Any]]:
        payload = self.get(f"/api/competitions/{self.competition}/my_teams")
        return list(payload.get("data") or [])

    def results(self, stage_id: int) -> List[Dict[str, Any]]:
        payload = self.get(
            f"/api/competitions/{self.competition}/results.json",
            params={
                "stage_id": stage_id,
                "module_type": "worksubmit",
                "page": 1,
                "limit": 9999999,
                "keyword": "",
            },
        )
        return list(payload.get("results") or [])

    def upload_token(self) -> Dict[str, Any]:
        payload = self.get("/api/buckets/get_upload_token.json")
        encrypted = base64.b64decode(payload["data"])
        try:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
            from cryptography.hazmat.primitives.padding import PKCS7
        except ImportError as exc:
            raise SubmissionError(
                "cryptography is required; install requirements-submit.txt"
            ) from exc

        decryptor = Cipher(
            algorithms.AES(TOKEN_AES_KEY), modes.CBC(TOKEN_AES_IV)
        ).decryptor()
        padded = decryptor.update(encrypted) + decryptor.finalize()
        unpadder = PKCS7(128).unpadder()
        decoded = unpadder.update(padded) + unpadder.finalize()
        token = json.loads(decoded.decode("utf-8"))
        required = {
            "access_key_id",
            "access_key_secret",
            "security_token",
            "end_point",
            "bucket",
            "callback_url",
            "bucket_host",
        }
        missing = sorted(required.difference(token))
        if missing:
            raise SubmissionError(f"upload token is missing fields: {missing}")
        return token


def _print_results(results: Iterable[Mapping[str, Any]]) -> None:
    rows = list(results)
    if not rows:
        print("existing submissions: none")
        return
    print("existing submissions:")
    for item in rows:
        score = item.get("data_ranking")
        score_text = "pending" if score is None else str(score)
        print(
            f"  id={item.get('id')} status={item.get('status')} "
            f"score={score_text} file={item.get('file_name')}"
        )


def _callback_headers(token: Mapping[str, Any], metadata: Mapping[str, Any]) -> Dict[str, str]:
    callback_body = (
        "bucket=${bucket}&object=${object}&etag=${etag}&size=${size}"
        "&mimeType=${mimeType}&my_var=${x:my_var}&"
        + _callback_query(metadata)
    )
    callback = {
        "callbackUrl": token["callback_url"],
        "callbackBody": callback_body,
        "callbackHost": token["bucket_host"],
    }
    encoded = base64.b64encode(
        json.dumps(callback, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
    return {"x-oss-callback": encoded}


def _read_oss_result(result: Any) -> bytes:
    # oss2 2.19 returns a PutObjectResult whose response stream is in `.resp`;
    # older/mocked clients may expose `.read()` directly.
    stream = getattr(result, "resp", result)
    reader = getattr(stream, "read", None)
    if not callable(reader):
        raise SubmissionError("OSS completion result did not expose a response body")
    return reader()


def upload_archive(
    path: Path,
    token: Mapping[str, Any],
    metadata: Mapping[str, Any],
    part_size_mib: int,
) -> Dict[str, Any]:
    try:
        import oss2
    except ImportError as exc:
        raise SubmissionError(
            "oss2 is required; install requirements-submit.txt"
        ) from exc

    auth = oss2.StsAuth(
        token["access_key_id"],
        token["access_key_secret"],
        token["security_token"],
    )
    bucket = oss2.Bucket(auth, token["end_point"], token["bucket"])
    object_name = str(uuid.uuid4())
    init = bucket.init_multipart_upload(
        object_name, headers={"Content-Type": "application/zip"}
    )
    upload_id = init.upload_id
    parts = []
    uploaded = 0
    total = path.stat().st_size
    part_size = part_size_mib * 1024 * 1024

    try:
        with path.open("rb") as stream:
            part_number = 1
            while True:
                block = stream.read(part_size)
                if not block:
                    break
                result = bucket.upload_part(
                    object_name, upload_id, part_number, block
                )
                parts.append(oss2.models.PartInfo(part_number, result.etag))
                uploaded += len(block)
                print(
                    f"uploaded: {uploaded}/{total} bytes ({uploaded / total:.1%})",
                    flush=True,
                )
                part_number += 1

        complete = bucket.complete_multipart_upload(
            object_name,
            upload_id,
            parts,
            headers=_callback_headers(token, metadata),
        )
        raw = _read_oss_result(complete)
    except Exception:
        try:
            bucket.abort_multipart_upload(object_name, upload_id)
        except Exception:
            pass
        raise

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CallbackUncertain(
            "upload callback returned a non-JSON response"
        ) from exc
    if payload.get("status") != 0:
        raise SubmissionError(
            f"upload callback failed: {payload.get('message', payload)}"
        )
    return payload


def _find_new_result(
    client: EducoderClient,
    stage_id: int,
    file_name: str,
    old_ids: set,
    attempts: int = 12,
) -> Optional[Dict[str, Any]]:
    for attempt in range(attempts):
        try:
            current_results = client.results(stage_id)
        except requests.RequestException:
            if attempt + 1 < attempts:
                time.sleep(5)
            continue
        matches = [
            item
            for item in current_results
            if item.get("file_name") == file_name and item.get("id") not in old_ids
        ]
        if matches:
            return max(matches, key=lambda item: item.get("id") or 0)
        if attempt + 1 < attempts:
            time.sleep(5)
    return None


def wait_for_result(
    client: EducoderClient,
    stage_id: int,
    result_id: int,
    timeout_seconds: float,
    poll_seconds: float,
) -> Dict[str, Any]:
    if timeout_seconds < 0:
        raise SubmissionError("--wait-timeout must be non-negative")
    if poll_seconds <= 0:
        raise SubmissionError("--poll-seconds must be positive")

    deadline = time.monotonic() + timeout_seconds
    last_state = None
    last_error = None
    while True:
        result = None
        try:
            results = client.results(stage_id)
        except requests.RequestException as exc:
            error_state = (type(exc).__name__, str(exc))
            if error_state != last_error:
                print(
                    f"result {result_id}: transient API error "
                    f"{type(exc).__name__}; retrying",
                    flush=True,
                )
                last_error = error_state
        else:
            last_error = None
            result = next(
                (item for item in results if item.get("id") == result_id),
                None,
            )
        if result is not None:
            state = (
                result.get("status"),
                result.get("data_ranking"),
                result.get("updated_at"),
            )
            if state != last_state:
                print(
                    f"result {result_id}: status={state[0]} "
                    f"score={state[1]} updated_at={state[2]}",
                    flush=True,
                )
                last_state = state
            if result.get("status") == 2:
                return result
            if result.get("status") == 3:
                message = result.get("err_msg") or "evaluation failed"
                raise SubmissionError(f"result {result_id} failed: {message}")

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            if result is None:
                raise SubmissionError(
                    f"result {result_id} was not found before timeout"
                )
            raise SubmissionError(
                f"result {result_id} did not finish within "
                f"{timeout_seconds:g} seconds"
            )
        time.sleep(min(poll_seconds, remaining))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_path", nargs="?", default=DEFAULT_ARCHIVE)
    parser.add_argument("--competition", default=DEFAULT_COMPETITION)
    parser.add_argument("--container-id", type=int, default=DEFAULT_CONTAINER_ID)
    parser.add_argument("--stage-id", type=int, default=DEFAULT_STAGE_ID)
    parser.add_argument("--team-id", type=int, default=DEFAULT_TEAM_ID)
    parser.add_argument("--expected-count", type=int, default=200)
    parser.add_argument("--expected-points", type=int, default=50000)
    parser.add_argument("--max-size-gib", type=float, default=1.0)
    parser.add_argument("--part-size-mib", type=int, default=8)
    parser.add_argument("--cookie-env", default="EDUCODER_COOKIE")
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="authenticate, print existing results, and exit without validating a ZIP",
    )
    parser.add_argument(
        "--wait-result-id",
        type=int,
        help="wait for an existing result ID instead of uploading a ZIP",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="after upload, wait until Educoder finishes evaluating the result",
    )
    parser.add_argument("--wait-timeout", type=float, default=1800.0)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--allow-duplicate", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.part_size_mib < 1:
        raise SubmissionError("--part-size-mib must be at least 1")
    cookie = os.environ.get(args.cookie_env, "")
    if not cookie:
        raise SubmissionError(
            f"set {args.cookie_env} in the environment; do not save the cookie in a file"
        )

    client = EducoderClient(cookie, args.competition)
    user = client.user_info()
    login = user.get("login")
    if not login:
        raise SubmissionError("authenticated user has no login")
    teams = client.teams()
    team = next((item for item in teams if item.get("id") == args.team_id), None)
    if team is None:
        raise SubmissionError(
            f"team {args.team_id} is not available to authenticated user {login}"
        )
    print(f"account OK: login={login} team={team.get('name')}({args.team_id})")

    before = client.results(args.stage_id)
    _print_results(before)
    if args.wait_result_id is not None:
        completed = wait_for_result(
            client,
            args.stage_id,
            args.wait_result_id,
            args.wait_timeout,
            args.poll_seconds,
        )
        print(
            f"evaluation complete: id={completed.get('id')} "
            f"score={completed.get('data_ranking')} "
            f"info={completed.get('err_msg')}"
        )
        return 0
    if args.status_only:
        return 0

    archive_path = Path(args.zip_path).expanduser().resolve()
    archive = validate_archive(
        archive_path,
        args.expected_count,
        args.expected_points,
        args.max_size_gib,
    )
    print(
        f"archive OK: {archive['file_name']} size={archive['size']} "
        f"sha256={archive['sha256']}"
    )

    duplicates = [
        item for item in before if item.get("file_name") == archive["file_name"]
    ]
    if duplicates and args.dry_run and not args.allow_duplicate:
        print(
            "dry-run warning: an existing submission has the same file name; "
            "a real upload would be rejected unless --allow-duplicate is passed"
        )
    elif duplicates and not args.allow_duplicate:
        raise SubmissionError(
            "an existing submission has the same file name; rename the archive or pass "
            "--allow-duplicate explicitly"
        )

    token = client.upload_token()
    print(
        "upload route OK: "
        f"bucket={token['bucket']} endpoint={token['end_point']} "
        f"callback_host={token['bucket_host']}"
    )
    if args.dry_run:
        print("dry-run complete; no file was uploaded and no result was changed")
        return 0

    if not args.yes:
        answer = input(
            f"Type SUBMIT to add {archive['file_name']} to stage {args.stage_id}: "
        )
        if answer != "SUBMIT":
            raise SubmissionError("submission cancelled")

    metadata = {
        "login": login,
        "container_type": "Competition",
        "file_name": archive["file_name"],
        "stage_type": args.stage_id,
        "container_id": args.container_id,
        "result_id": None,
        "module_type": "worksubmit",
        "competition_team_id": args.team_id,
    }
    old_ids = {item.get("id") for item in before}
    callback_uncertain = False
    try:
        response = upload_archive(
            archive_path, token, metadata, args.part_size_mib
        )
    except CallbackUncertain as exc:
        # Educoder can create the result successfully while its OSS callback
        # returns an HTML/empty body.  The result list is authoritative, so do
        # not upload the same archive again before checking it.
        callback_uncertain = True
        print(f"callback response uncertain: {exc}; checking result list")
    else:
        print(f"callback accepted: status={response.get('status')}")
    created = _find_new_result(
        client, args.stage_id, archive["file_name"], old_ids
    )
    if created is None:
        if callback_uncertain:
            raise SubmissionError(
                "upload completed with an uncertain callback, and no new result "
                "was visible after 60 seconds; inspect the competition page before retrying"
            )
        raise SubmissionError(
            "callback succeeded, but the new result was not visible after 60 seconds"
        )
    print(
        f"submission created: id={created.get('id')} status={created.get('status')} "
        f"score={created.get('data_ranking')} file={created.get('file_name')}"
    )
    if args.wait:
        completed = wait_for_result(
            client,
            args.stage_id,
            int(created["id"]),
            args.wait_timeout,
            args.poll_seconds,
        )
        print(
            f"evaluation complete: id={completed.get('id')} "
            f"score={completed.get('data_ranking')} "
            f"info={completed.get('err_msg')}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SubmissionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
