#!/usr/bin/env python3
"""Unified NKAI Track2 entrypoint for environment checks and reproduction stages."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from plr3d.backend import add_device_argument, configure_device, device_status


ROOT = Path(__file__).resolve().parent


def prepare_conda_cuda_layout() -> None:
    """Make conda CUDA Toolkit visible to Jittor 1.3.11 without root writes."""
    prefix = Path(sys.prefix)
    nvcc = prefix / "bin" / "nvcc"
    cudart = prefix / "lib" / "libcudart.so"
    lib64 = prefix / "lib64"
    if not nvcc.is_file() or not cudart.is_file():
        return
    os.environ["PATH"] = str(nvcc.parent) + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("CUDA_HOME", str(prefix))
    if not lib64.exists():
        try:
            lib64.symlink_to("lib", target_is_directory=True)
        except FileExistsError:
            pass
ENTRYPOINTS: Dict[str, str] = {
    "prepare-keys": "tools/build_key_list.py",
    "filter-keys": "tools/filter_key_list.py",
    "prepare-data": "tools/prepare_reproduction_data.py",
    "train": "train.py",
    "train-rot": "train_rot.py",
    "infer": "infer.py",
    "generate-rot": "generate_rot.py",
    "assemble-incumbent": "assemble_incumbent.py",
    "assemble-full": "assemble_full.py",
    "assemble-plr003": "assemble.py",
    "run-plr003": "run_plr003.py",
    "package-results": "package_results.py",
    "audit-source": "tools/audit_source_archive.py",
}


def run_command(command: List[str], cwd: Path, log_path: Optional[Path] = None) -> None:
    print("+", " ".join(command), flush=True)
    if log_path is None:
        subprocess.run(command, cwd=str(cwd), check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            cwd=str(cwd),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if process.returncode != 0:
        raise RuntimeError(f"stage failed with code {process.returncode}; see {log_path}")


def doctor(device: str, deep: bool) -> int:
    import jittor as jt

    configure_device(device, jt)
    import numpy as np

    python_version = tuple(sys.version_info[:3])
    jittor_version = tuple(int(item) for item in jt.__version__.split(".")[:3])
    status = device_status(device, jt)
    checks = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "python_3_10_or_newer": python_version >= (3, 10, 0),
        "jittor": jt.__version__,
        "jittor_1_3_10_or_newer": jittor_version >= (1, 3, 10),
        "numpy": np.__version__,
        "cuda_available": bool(jt.compiler.has_cuda),
        "nvcc_path": str(getattr(jt.compiler, "nvcc_path", "")),
        "device_status": status,
    }
    checks.update(status)
    print(json.dumps(checks, indent=2, sort_keys=True))
    if not checks["python_3_10_or_newer"] or not checks["jittor_1_3_10_or_newer"]:
        return 2
    if deep:
        run_command(
            [
                sys.executable,
                str(ROOT / "tests" / "smoke_jittor.py"),
                "--device",
                device,
            ],
            ROOT,
        )
    return 0


def delegate(name: str, arguments: Sequence[str]) -> int:
    target = ROOT / ENTRYPOINTS[name]
    run_command([sys.executable, str(target), *arguments], ROOT)
    return 0


def load_pipeline(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != 1 or not isinstance(payload.get("stages"), list):
        raise ValueError("pipeline must contain version=1 and a stages list")
    return payload


def expand(value: str) -> str:
    return os.path.expanduser(os.path.expandvars(value))


def run_pipeline(config: Path, log_dir: Path, dry_run: bool) -> int:
    payload = load_pipeline(config)
    log_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for index, stage in enumerate(payload["stages"], 1):
        name = str(stage.get("name", f"stage-{index:02d}"))
        entrypoint = str(stage["entrypoint"])
        if entrypoint not in ENTRYPOINTS:
            raise ValueError(f"unknown pipeline entrypoint: {entrypoint}")
        arguments = [expand(str(item)) for item in stage.get("args", [])]
        skip_path = stage.get("skip_if_exists")
        if skip_path and Path(expand(str(skip_path))).exists():
            print(f"SKIP {name}: {expand(str(skip_path))} exists", flush=True)
            summary.append({"name": name, "status": "skipped"})
            continue
        command = [sys.executable, str(ROOT / ENTRYPOINTS[entrypoint]), *arguments]
        if dry_run:
            print("DRY-RUN", name, "+", " ".join(command), flush=True)
            summary.append({"name": name, "status": "dry-run"})
            continue
        run_command(command, ROOT, log_dir / f"{index:02d}_{name}.log")
        summary.append({"name": name, "status": "completed"})
    (log_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser("doctor", help="check the Jittor environment")
    add_device_argument(doctor_parser)
    doctor_parser.add_argument(
        "--deep",
        action="store_true",
        help="run selected-device forward/backward smoke",
    )

    for name in ENTRYPOINTS:
        subparser = subparsers.add_parser(name, help=f"delegate to {ENTRYPOINTS[name]}")
        subparser.add_argument("arguments", nargs=argparse.REMAINDER)

    pipeline_parser = subparsers.add_parser("pipeline", help="run a resumable JSON stage pipeline")
    pipeline_parser.add_argument("--config", type=Path, required=True)
    pipeline_parser.add_argument("--log-dir", type=Path, required=True)
    pipeline_parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.command == "doctor":
        if args.device == "cuda":
            prepare_conda_cuda_layout()
        return doctor(args.device, args.deep)
    prepare_conda_cuda_layout()
    if args.command == "pipeline":
        return run_pipeline(args.config.resolve(), args.log_dir.resolve(), args.dry_run)
    arguments = list(args.arguments)
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    return delegate(args.command, arguments)


if __name__ == "__main__":
    raise SystemExit(main())
