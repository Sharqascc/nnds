"""Integration (heavy): CLI dispatches to every detector path.

Marked `integration` — runs nightly, not in PR CI. Requires LFS video
and detector weights. Skips gracefully if assets are absent.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
VIDEO = REPO / "data" / "sample_data" / "anonymized_traffic_video_50f.mp4"
UVH = REPO / "data" / "models" / "uvh26.pt"
YOLO = REPO / "data" / "models" / "yolo11n.pt"

pytestmark = pytest.mark.integration


def _skip_if_missing(paths):
    for p in paths:
        if not Path(p).exists() or Path(p).stat().st_size < 1_000_000:
            pytest.skip(f"asset missing or pointer: {p}")


def _run(detector: str, extra: list[str]) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable, "-m", "src.pipeline.traffic_analyzer",
        "--video", str(VIDEO),
        "--out-csv", f"/tmp/int_{detector}.csv",
        "--detector", detector,
        "--device", "cpu",
        "--max-frames", "20",
    ] + extra
    return subprocess.run(cmd, cwd=REPO, capture_output=True,
                          text=True, timeout=600)


def test_uvh_coco_fused_dispatch_runs():
    _skip_if_missing([VIDEO, UVH, YOLO])
    r = _run("uvh-coco-fused", [
        "--uvh-model", str(UVH),
        "--yolo-weights", str(YOLO),
        "--coco-person-model", str(YOLO),
    ])
    assert r.returncode == 0, r.stderr[-1500:]


def test_yolo_cpu_dispatch_runs():
    _skip_if_missing([VIDEO, YOLO])
    r = _run("yolo-cpu", ["--yolo-weights", str(YOLO)])
    assert r.returncode == 0, r.stderr[-1500:]


def test_unknown_detector_is_rejected():
    r = _run("this-is-not-a-detector", [])
    assert r.returncode != 0
    assert "Unsupported detector" in (r.stderr + r.stdout)
