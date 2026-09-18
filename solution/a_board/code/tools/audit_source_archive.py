#!/usr/bin/env python3
"""Audit a code-only PLR source tree or ZIP before review submission."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import zipfile
from pathlib import Path, PurePosixPath


FORBIDDEN_SUFFIXES = {
    ".pth",
    ".pt",
    ".ckpt",
    ".pkl",
    ".npy",
    ".npz",
    ".obj",
    ".ply",
    ".zip",
    ".tar",
    ".gz",
    ".log",
    ".pyc",
}
PRIVATE_PATTERNS = [
    re.compile(r"/root/data-tmp/", re.IGNORECASE),
    re.compile(r"[A-Z]:[/\\]Users[/\\]", re.IGNORECASE),
    re.compile(r"BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
]
PROHIBITED_RUNTIME_MODULES = (
    "torch",
    "triton",
    "mamba_ssm",
    "causal_conv1d",
    "scipy",
    "trimesh",
    "omegaconf",
    "tqdm",
)
PROHIBITED_IMPORT = re.compile(
    r"^\s*(?:from\s+(?:" + "|".join(PROHIBITED_RUNTIME_MODULES) + r")\b"
    r"|import\s+(?:" + "|".join(PROHIBITED_RUNTIME_MODULES) + r")\b)",
    re.MULTILINE,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_tree(root: Path):
    errors = []
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if "__pycache__" in path.parts or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden artifact: {relative}")
            continue
        if path.stat().st_size > 5_000_000:
            errors.append(f"unexpected large source file: {relative} ({path.stat().st_size})")
        if path.suffix.lower() in {".py", ".md", ".txt", ".tex", ".json", ".yaml", ".yml", ".sh"}:
            text = path.read_text(encoding="utf-8")
            if relative != "tools/audit_source_archive.py":
                for pattern in PRIVATE_PATTERNS:
                    if pattern.search(text):
                        errors.append(
                            f"private path/secret pattern in {relative}: {pattern.pattern}"
                        )
            if path.suffix.lower() == ".py":
                try:
                    ast.parse(text, filename=relative, feature_version=8)
                except SyntaxError as exc:
                    errors.append(f"Python 3.8 syntax error in {relative}: {exc}")
                if PROHIBITED_IMPORT.search(text):
                    errors.append(f"prohibited runtime import: {relative}")
        files.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return errors, files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--zip", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    errors, files = audit_tree(root)
    zip_record = None
    if args.zip is not None:
        archive = args.zip.resolve()
        if not archive.is_file():
            errors.append(f"missing ZIP: {archive}")
        else:
            size = archive.stat().st_size
            if size >= 100_000_000:
                errors.append(f"ZIP is not below 100,000,000 bytes: {size}")
            top_level = None
            with zipfile.ZipFile(archive) as handle:
                names = handle.namelist()
                if len(names) != len(set(names)):
                    errors.append("ZIP contains duplicate member names")
                if handle.testzip() is not None:
                    errors.append("ZIP CRC test failed")
                top_levels = set()
                for name in names:
                    parts = PurePosixPath(name).parts
                    if len(parts) < 2 or name.startswith("/") or ".." in parts:
                        errors.append(f"unsafe or top-level-free ZIP member: {name}")
                        continue
                    top_levels.add(parts[0])
                    suffix = PurePosixPath(name).suffix.lower()
                    if suffix in FORBIDDEN_SUFFIXES - {".zip"}:
                        errors.append(f"forbidden ZIP member: {name}")
                if len(top_levels) != 1:
                    errors.append(f"ZIP must have exactly one top-level directory: {top_levels}")
                else:
                    top_level = next(iter(top_levels))
                    expected_names = {
                        f"{top_level}/{record['path']}" for record in files
                    }
                    if set(names) != expected_names:
                        missing = sorted(expected_names - set(names))
                        extra = sorted(set(names) - expected_names)
                        errors.append(
                            f"ZIP/tree member mismatch: missing={missing[:3]} extra={extra[:3]}"
                        )
                    for record in files:
                        name = f"{top_level}/{record['path']}"
                        if name in names and hashlib.sha256(handle.read(name)).hexdigest() != record["sha256"]:
                            errors.append(f"ZIP/tree content mismatch: {name}")
            zip_record = {
                "path": str(archive),
                "size": size,
                "below_100000000_bytes": size < 100_000_000,
                "below_100_mib": size < 100 * 1024 * 1024,
                "sha256": sha256(archive),
                "member_count": len(names),
                "top_level": top_level,
            }
    payload = {
        "audit_pass": not errors,
        "root": str(root),
        "file_count": len(files),
        "total_source_bytes": sum(item["size"] for item in files),
        "zip": zip_record,
        "errors": errors,
        "files": files,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("audit_pass", "file_count", "total_source_bytes", "errors")}, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
