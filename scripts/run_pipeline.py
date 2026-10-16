#!/usr/bin/env python3
"""
Main entry point for the NNDS Intersection Safety Pipeline.

Prepares model weights, applies the global seed, and dispatches to the
analyzer in src/pipeline/traffic_analyzer.py (or the installed
`pipeline.traffic_analyzer` module if this repository is pip-installed).

Usage:
    python scripts/run_pipeline.py --video path/to/video.mp4 [options]
    python scripts/run_pipeline.py --video path/to/video.mp4 --skip-ensure

Or, after `pip install -e .`, as a console script:

    nnds-run --video path/to/video.mp4 [options]
"""

from __future__ import annotations

import importlib.util
import os
import runpy
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Flags that do not require model weights. Prep is skipped for these to avoid
# downloading ~100 MB of weights just to print a version string.
_INFORMATIONAL_FLAGS = {
    "-h",
    "--help",
    "--version",
    "--self-check",
    "--dry-run",
    "--demo",
}

# Runner-only flags that must not be forwarded to the analyzer's argparse.
_RUNNER_ONLY_FLAGS = {"--skip-ensure", "--auto-time-of-day"}


def _has_informational_flag(argv: list[str]) -> bool:
    return any(a in _INFORMATIONAL_FLAGS for a in argv)


def ensure_models_if_needed() -> None:
    """Run scripts/ensure_models.py unless skipped or unneeded.

    Reads sys.argv directly because the final dispatch to the pipeline uses
    runpy and does not go through argparse in this module.
    """
    if "--skip-ensure" in sys.argv:
        sys.argv.remove("--skip-ensure")
        return

    if _has_informational_flag(sys.argv):
        return

    ensure_script = REPO_ROOT / "scripts" / "ensure_models.py"
    if not ensure_script.exists():
        # Installed-package context: no checkout, nothing to prepare here.
        return

    print("\n[run_pipeline] Checking models...")
    # Run in the repository root so the exporter's relative weight paths
    # resolve consistently regardless of the caller's cwd.
    result = subprocess.run(
        [sys.executable, str(ensure_script), "--imgsz", "640"],
        cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        print(f"[run_pipeline] Model preparation failed (exit {result.returncode}).")
        sys.exit(result.returncode)


def _maybe_estimate_time_of_day() -> None:
    """If --auto-time-of-day is set, estimate a label and forward it.

    Runner-only flags (`--auto-time-of-day`) are stripped from sys.argv before
    dispatch so the analyzer's argparse does not reject them. The estimated
    label is forwarded via `--time-of-day`, which the analyzer accepts.
    """
    if "--auto-time-of-day" not in sys.argv:
        return

    # Strip the runner-only flag first; it must never reach the analyzer.
    while "--auto-time-of-day" in sys.argv:
        sys.argv.remove("--auto-time-of-day")

    # User supplied an explicit label; don't override.
    if "--time-of-day" in sys.argv:
        return

    video_arg_idx = None
    for i, arg in enumerate(sys.argv):
        if arg == "--video" and i + 1 < len(sys.argv):
            video_arg_idx = i + 1
            break
    if video_arg_idx is None:
        print("[run_pipeline] --video not found; cannot auto-estimate time-of-day.")
        return

    video_path = sys.argv[video_arg_idx]
    estimator = REPO_ROOT / "scripts" / "estimate_time_of_day.py"
    if not estimator.exists():
        print(f"[run_pipeline] estimator not found at {estimator}; skipping.")
        return

    print("\n[run_pipeline] Estimating time-of-day via VLM...")
    result = subprocess.run(
        [sys.executable, str(estimator), "--video", video_path],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(
            f"[run_pipeline] estimator failed (exit {result.returncode}); "
            f"continuing without a label."
        )
        print(result.stderr[-500:])
        return

    label = result.stdout.strip()
    if not label:
        print("[run_pipeline] estimator produced no label; continuing without one.")
        return

    print(f"[run_pipeline] Estimated time-of-day: {label}")
    sys.argv.extend(["--time-of-day", label])


def _analyzer_spec_available() -> bool:
    """Return True iff `pipeline.traffic_analyzer` is importable as a module.

    `importlib.util.find_spec` imports the parent packages first, so it raises
    ModuleNotFoundError when `pipeline` itself is not on sys.path (e.g. a source
    checkout without `pip install -e .`). That is a legitimate "not installed"
    answer, not an error, so we translate it to False.
    """
    try:
        return importlib.util.find_spec("pipeline.traffic_analyzer") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _dispatch_to_analyzer() -> None:
    """Run the analyzer as __main__, whether installed or in a source tree."""
    # Installed package: `pip install .` installs it as `pipeline.traffic_analyzer`.
    if _analyzer_spec_available():
        sys.argv[0] = "pipeline.traffic_analyzer"
        runpy.run_module("pipeline.traffic_analyzer", run_name="__main__", alter_sys=False)
        return

    # Source tree: `src/pipeline/traffic_analyzer.py`.
    pipeline_path = REPO_ROOT / "src" / "pipeline" / "traffic_analyzer.py"
    if pipeline_path.exists():
        sys.argv[0] = str(pipeline_path)
        runpy.run_path(str(pipeline_path), run_name="__main__")
        return

    print(
        "[run_pipeline] ERROR: could not locate traffic_analyzer in installed "
        "packages or source tree.",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> None:
    """Prepare models, apply seeding, and dispatch to the pipeline module.

    Invoked by the `nnds-run` console script and by
    `python scripts/run_pipeline.py`.
    """
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from src.utils.seed import set_seed

    # NNDS_SEED env var overrides the default. set_seed exports PYTHONHASHSEED
    # so subprocesses inherit it.
    raw_seed = os.environ.get("NNDS_SEED", "42")
    try:
        _seed = int(raw_seed)
    except ValueError:
        raise SystemExit(f"NNDS_SEED must be an integer, got {raw_seed!r}") from None
    try:
        set_seed(_seed)
    except ValueError as e:
        raise SystemExit(f"invalid NNDS_SEED: {e}") from None

    ensure_models_if_needed()
    _maybe_estimate_time_of_day()

    _dispatch_to_analyzer()


if __name__ == "__main__":
    main()
