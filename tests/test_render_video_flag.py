"""Tests for the --out-video flag and render_full_video."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def test_out_video_flag_registered():
    from src.pipeline.traffic_analyzer import parse_args

    args = parse_args(["--out-video", "/tmp/x.mp4"])
    assert args.out_video == "/tmp/x.mp4"


def test_out_video_default_is_none():
    from src.pipeline.traffic_analyzer import parse_args

    args = parse_args([])
    assert args.out_video is None


def test_out_video_help_text_explains_render():
    from src.pipeline.traffic_analyzer import _build_parser

    text = _build_parser().format_help().lower()
    assert "--out-video" in text
    assert any(k in text for k in ("annotated", "render", "overlay"))


def test_render_video_helper_called_when_flag_set(monkeypatch, tmp_path):
    """_render_video is invoked only when --out-video is set."""
    from src.pipeline import traffic_analyzer as ta

    called = {"rendered": False}

    def fake_render(args):
        called["rendered"] = True

    def fake_run_pipeline(args):
        return None

    monkeypatch.setattr(ta, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(ta, "_render_video", fake_render)

    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")

    # with --out-video
    monkeypatch.setattr(
        "sys.argv", ["nnds", "--video", str(video), "--out-video", str(tmp_path / "o.mp4")]
    )
    ta.main()
    assert called["rendered"] is True

    # without --out-video
    called["rendered"] = False
    monkeypatch.setattr("sys.argv", ["nnds", "--video", str(video)])
    ta.main()
    assert called["rendered"] is False


@pytest.mark.slow
def test_render_full_video_creates_mp4(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    from src.analysis.visualization.render_full_video import render_full_video

    vid = tmp_path / "in.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(vid), fourcc, 10.0, (100, 100))
    for _ in range(5):
        writer.write(np.zeros((100, 100, 3), dtype=np.uint8))
    writer.release()

    events_csv = tmp_path / "events.csv"
    with events_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pet", "frame", "track_a", "track_b", "conflict_type"])

    out = tmp_path / "out.mp4"
    summary = render_full_video(
        video_path=str(vid),
        events_csv=str(events_csv),
        out_path=str(out),
        max_frames=5,
    )
    assert out.exists() and out.stat().st_size > 0
    assert summary["frames_written"] == 5
    assert summary["events_drawn"] == 0
