#!/usr/bin/env python3
"""Reproducible main entry point for NNDS in Colab."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parent

PROJECT_NAME = "nnds"
PROJECT_MODULE = "pipeline"
CLI_MODULE = "pipeline.traffic_analyzer"
CLI_COMMAND = "nnds"

RUNTIME_REQUIREMENTS = (
    "torch>=2.14.1,<2.15",
    "torchvision>=0.29.1,<0.30",
    "ultralytics>=8.4.178,<8.5",
    "opencv-python>=5.0.0,<5.1",
    "numpy>=2.5.4,<2.6",
    "pandas>=3.0.6,<3.1",
    "scipy>=1.18.1,<1.19",
    "matplotlib>=3.11.2,<3.12",
    "tqdm>=4.70.1,<4.71",
    "PyYAML>=6.0.3,<6.1",
    "openvino>=2026.4.1,<2026.5",
    "lap>=0.5.13,<0.6",
    "scikit-learn>=1.9.1,<1.10",
)


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _importable(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def install_requirements(*, install_dev: bool = False) -> None:
    extras = ".[dev]" if install_dev else "."

    print(f"[1/5] Installing requirements: pip install -e {extras}")

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
        cwd=REPO_ROOT,
        check=True,
    )

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", extras],
        cwd=REPO_ROOT,
        check=True,
    )


def verify_installation() -> None:
    print("[2/5] Verifying installation")

    missing = []

    for requirement in RUNTIME_REQUIREMENTS:
        package = requirement.split(">=")[0].strip()

        if _version(package) is None:
            missing.append(requirement)

    if missing:
        raise RuntimeError(
            "Missing runtime requirements: " + ", ".join(missing)
        )

    if _version(PROJECT_NAME) is None:
        raise RuntimeError(f"Project not installed: {PROJECT_NAME}")

    if not _importable(PROJECT_MODULE):
        raise RuntimeError(f"Module not importable: {PROJECT_MODULE}")

    if not _importable(CLI_MODULE):
        raise RuntimeError(f"CLI module not importable: {CLI_MODULE}")

    print(
        f"      Verified {PROJECT_NAME} {_version(PROJECT_NAME)}; "
        "all runtime requirements installed."
    )


def run_main_task(forwarded_args: Sequence[str]) -> None:
    print("[3/5] Running nnds")

    if forwarded_args and forwarded_args[0] == "--":
        forwarded_args = forwarded_args[1:]

    if not forwarded_args:
        print(
            "      No task arguments supplied; built-in demo will run in "
            "the remaining-workflow stage."
        )
        return

    command = [
        sys.executable,
        "-m",
        CLI_MODULE,
        *forwarded_args,
    ]

    print("      " + " ".join(command))

    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)

    if completed.returncode != 0:
        raise RuntimeError(
            f"nnds failed with exit code {completed.returncode}."
        )


def run_remaining_tasks() -> None:
    print("[4/5] Running remaining project workflow stages")

    demo_command = [
        sys.executable,
        "-m",
        CLI_MODULE,
        "--demo",
    ]

    print("      Running built-in NNDS demo:", " ".join(demo_command))

    demo = subprocess.run(demo_command, cwd=REPO_ROOT, check=False)

    if demo.returncode != 0:
        raise RuntimeError(
            f"Built-in NNDS demo failed with exit code {demo.returncode}."
        )

    print("      Built-in NNDS demo completed successfully.")


def write_run_manifest() -> None:
    print("[5/5] Writing run manifest")

    manifest_path = REPO_ROOT / "outputs" / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "project": PROJECT_NAME,
        "project_version": _version(PROJECT_NAME),
        "python": sys.version,
        "cli_command": CLI_COMMAND,
        "cli_module": CLI_MODULE,
    }

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"      Wrote {manifest_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_colab.py",
        description=(
            "Reproducible Colab entry point for NNDS. Installs requirements, "
            "verifies the environment, runs the nnds CLI, runs its built-in "
            "demo, and writes a run manifest."
        ),
    )

    parser.add_argument("--install-dev", action="store_true")
    parser.add_argument(
        "cli_args",
        nargs="*",
        help="Arguments forwarded directly to the nnds CLI.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    src_root = REPO_ROOT / "src"

    for path in (str(REPO_ROOT), str(src_root)):
        if path not in sys.path:
            sys.path.insert(0, path)

    os.environ["PYTHONPATH"] = os.pathsep.join(
        path
        for path in (str(src_root), os.environ.get("PYTHONPATH", ""))
        if path
    )

    os.chdir(REPO_ROOT)

    install_requirements(install_dev=args.install_dev)
    verify_installation()
    run_main_task(args.cli_args)
    run_remaining_tasks()
    write_run_manifest()

    print("[done] Main entry-point workflow completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
