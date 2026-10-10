"""Regression guard for issue #46.

Before the fix, reproduce_pipeline.sh printed "Reproduction complete" and
exited 0 even when no output CSV was ever written.

These tests exercise the preflight path only (via --preflight-only), so
they run in seconds and need no models or real videos.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "reproduce_pipeline.sh"
MIN_VIDEO_BYTES = 1_048_576


def _run_preflight(video_dir: Path, *, giti_only: bool = False):
    env = os.environ.copy()
    env["NNDS_VIDEO_DIR"] = str(video_dir)
    args = ["bash", str(SCRIPT), "--preflight-only"]
    if giti_only:
        args.append("--giti-only")
    return subprocess.run(args, capture_output=True, text=True, env=env, cwd=str(REPO))


def _fake_video(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * size)


def test_missing_videos_exit_nonzero(tmp_path):
    """Issue #46 case 1: both videos missing -> must not exit 0."""
    r = _run_preflight(tmp_path)
    assert r.returncode != 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert "git lfs pull" in (r.stdout + r.stderr)


def test_lfs_pointer_sized_videos_exit_nonzero(tmp_path):
    """Videos present but tiny (LFS pointers) -> nonzero + clear message."""
    _fake_video(tmp_path / "GITI_traffic_video.mp4", 132)
    _fake_video(tmp_path / "MRC_traffic_video.mp4", 132)
    r = _run_preflight(tmp_path)
    assert r.returncode != 0, r.stdout + r.stderr
    assert "git lfs pull" in (r.stdout + r.stderr)


def test_real_sized_videos_exit_zero(tmp_path):
    """Sanity: with valid-looking videos, preflight passes."""
    _fake_video(tmp_path / "GITI_traffic_video.mp4", MIN_VIDEO_BYTES + 1)
    _fake_video(tmp_path / "MRC_traffic_video.mp4", MIN_VIDEO_BYTES + 1)
    r = _run_preflight(tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Preflight OK" in r.stdout


def test_giti_only_ignores_missing_mrc(tmp_path):
    """--giti-only lets a GITI-only reproduction succeed with no MRC video."""
    _fake_video(tmp_path / "GITI_traffic_video.mp4", MIN_VIDEO_BYTES + 1)
    r = _run_preflight(tmp_path, giti_only=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_giti_only_still_fails_on_missing_giti(tmp_path):
    """--giti-only does not excuse a missing GITI video."""
    _fake_video(tmp_path / "MRC_traffic_video.mp4", MIN_VIDEO_BYTES + 1)
    r = _run_preflight(tmp_path, giti_only=True)
    assert r.returncode != 0, r.stdout + r.stderr
