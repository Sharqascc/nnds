#!/usr/bin/env python3
"""Reproducible one-command bootstrap for NNDS in Colab.

Usage:
    python scripts/bootstrap_colab.py

Branch selection:
    python scripts/bootstrap_colab.py --branch feature-name
    NNDS_BRANCH=feature-name python scripts/bootstrap_colab.py

Optional:
    python scripts/bootstrap_colab.py --skip-install
    python scripts/bootstrap_colab.py --skip-models
    python scripts/bootstrap_colab.py --skip-self-check
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

REPO_URL = "https://github.com/Sharqascc/nnds.git"
DEFAULT_BRANCH = "uvh-only-mep"
REPO_DIR = Path("/content/nnds")


def run(command: list[str], cwd: Path | None = None) -> None:
    print(f"\n$ {' '.join(command)}")
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
    )

    if result.returncode != 0:
        raise SystemExit(result.returncode)


def token_from_colab() -> str | None:
    try:
        from google.colab import userdata

        return userdata.get("GH_TOKEN")
    except Exception:
        return None


def authenticated_url() -> str:
    token = os.environ.get("GH_TOKEN") or token_from_colab()

    if not token:
        return REPO_URL

    parsed = urlparse(REPO_URL)
    return f"https://x-access-token:{token}@{parsed.netloc}{parsed.path}"


def clone_or_update(branch: str) -> None:
    if REPO_DIR.exists():
        run(["git", "fetch", "origin", branch], REPO_DIR)
        run(["git", "checkout", branch], REPO_DIR)
        run(
            ["git", "reset", "--hard", f"origin/{branch}"],
            REPO_DIR,
        )
        return

    run(
        [
            "git",
            "clone",
            "--branch",
            branch,
            authenticated_url(),
            str(REPO_DIR),
        ],
        Path("/content"),
    )

    run(
        [
            "git",
            "remote",
            "set-url",
            "origin",
            REPO_URL,
        ],
        REPO_DIR,
    )

    run(
        [
            "git",
            "branch",
            "--set-upstream-to=origin/" + branch,
            branch,
        ],
        REPO_DIR,
    )


def install_package() -> None:
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-e",
            ".[dev]",
        ],
        REPO_DIR,
    )


def ensure_models() -> None:
    run(
        [
            "bash",
            "scripts/download_models.sh",
        ],
        REPO_DIR,
    )

    run(
        [
            sys.executable,
            "scripts/ensure_models.py",
            "--skip-openvino",
        ],
        REPO_DIR,
    )


def self_check() -> None:
    run(
        [
            "nnds",
            "--self-check",
        ],
        REPO_DIR,
    )


def run_checks() -> None:
    """Run the same local quality gates used by CI."""

    run(
        [
            sys.executable,
            "-m",
            "compileall",
            "-q",
            "src",
            "tests",
            "scripts",
            "conftest.py",
        ],
        REPO_DIR,
    )

    run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            ".",
        ],
        REPO_DIR,
    )

    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--tb=short",
        ],
        REPO_DIR,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=("Reproducibly prepare and verify NNDS in Colab."))

    parser.add_argument(
        "--branch",
        default=os.environ.get("NNDS_BRANCH", DEFAULT_BRANCH),
        help=(f"Branch to clone or update. Defaults to NNDS_BRANCH, then {DEFAULT_BRANCH}."),
    )

    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip editable installation.",
    )

    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip model download and validation.",
    )

    parser.add_argument(
        "--skip-self-check",
        action="store_true",
        help="Skip nnds --self-check.",
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help=("Run the CI-equivalent compile, Ruff, and pytest checks after bootstrap."),
    )

    args = parser.parse_args()

    clone_or_update(args.branch)

    if not args.skip_install:
        install_package()

    if not args.skip_models:
        ensure_models()

    if not args.skip_self_check:
        self_check()

    if args.check:
        run_checks()

    print("\nNNDS bootstrap complete.")


if __name__ == "__main__":
    main()
