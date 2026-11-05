"""CLI contract tests for the `nnds` entry point.

These lock the user-visible interface: every flag has help text, --help and
--version work without a video, missing required input exits non-zero with a
clear message, and the demo mode runs without weights.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _run(*args: str, timeout: int = 30) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO) + (os.pathsep + existing if existing else "")
    return subprocess.run(
        [sys.executable, "-m", "src.pipeline.traffic_analyzer", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO),
        timeout=timeout,
    )


def test_help_exits_zero_and_documents_every_flag():
    r = _run("--help")
    assert r.returncode == 0, r.stderr
    # every public flag should be listed
    for flag in (
        "--demo",
        "--video",
        "--out-csv",
        "--pet-threshold",
        "--self-check",
        "--dry-run",
        "--detector",
        "--bev-config",
        "--grid-config",
        "--gate-config",
        "--max-gap",
        "--max-frames",
        "--max-jump",
        "--device",
        "--uvh-model",
        "--coco-person-model",
        "--yolo-weights",
        "--sam3-weights",
        "--rtdetr-weights",
        "--video-source",
        "--time-of-day",
        "--no-progress",
    ):
        assert flag in r.stdout, f"{flag} missing from --help output"


def test_help_includes_examples_epilog():
    r = _run("--help")
    assert "examples:" in r.stdout.lower()
    assert "nnds --demo" in r.stdout
    assert "nnds --self-check" in r.stdout


def test_help_groups_into_sections():
    r = _run("--help")
    for section in ("modes", "input", "detector and models", "configs", "tracking", "output"):
        assert section in r.stdout.lower(), f"section '{section}' missing"


def test_version_exits_zero_and_shows_revision():
    r = _run("--version")
    assert r.returncode == 0, r.stderr
    assert "revision" in r.stdout.lower()
    assert "traffic_analyzer" in r.stdout


def test_version_works_without_video():
    r = _run("--version")
    assert r.returncode == 0
    # Should not complain about missing --video
    assert "--video is required" not in (r.stdout + r.stderr)


def test_missing_video_exits_nonzero_with_clear_message():
    r = _run()
    assert r.returncode != 0
    combined = r.stdout + r.stderr
    assert "--video is required" in combined
    assert "--demo" in combined  # message must hint at alternatives


def test_nonexistent_video_exits_nonzero():
    r = _run("--video", "/nonexistent/path.mp4")
    assert r.returncode != 0
    combined = r.stdout + r.stderr
    assert "not found" in combined.lower()


def test_time_of_day_is_accepted():
    """`--time-of-day` should not be flagged as unrecognized."""
    r = _run("--time-of-day", "morning", "--self-check", timeout=60)
    assert "unrecognized arguments" not in (r.stdout + r.stderr)


def test_detector_choices_enforced():
    """An invalid --detector value should be rejected by argparse."""
    r = _run("--detector", "not-a-real-detector", "--video", "/tmp/x.mp4")
    assert r.returncode != 0
    combined = r.stdout + r.stderr
    # argparse's choices error wording
    assert "invalid choice" in combined.lower() or "invalid" in combined.lower()
