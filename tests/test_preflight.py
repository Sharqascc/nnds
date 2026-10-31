"""Tests for src.pipeline.preflight."""

from __future__ import annotations

from pathlib import Path

from src.pipeline import preflight


def test_missing_video_is_flagged():
    r = preflight.run_checks(video="/nonexistent/path.mp4")
    assert not r.ok
    assert any("video file" in p for p in r.problems)


def test_tiny_video_flagged_as_lfs_pointer(tmp_path):
    v = tmp_path / "x.mp4"
    v.write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
    r = preflight.run_checks(video=str(v))
    assert not r.ok
    assert any("LFS" in p or "lfs" in p for p in r.problems)


def test_valid_video_passes(tmp_path):
    v = tmp_path / "x.mp4"
    v.write_bytes(b"x" * (preflight.MIN_VIDEO_BYTES + 1))
    r = preflight.run_checks(video=str(v), out_csv=str(tmp_path / "out.csv"))
    video_checks = [c for c in r.checks if c["check"] == "video file"]
    assert video_checks and video_checks[0]["ok"]


def test_missing_weight_flagged(tmp_path):
    v = tmp_path / "x.mp4"
    v.write_bytes(b"x" * (preflight.MIN_VIDEO_BYTES + 1))
    r = preflight.run_checks(
        video=str(v),
        uvh_model="/nope/uvh26.pt",
        out_csv=str(tmp_path / "out.csv"),
    )
    assert not r.ok
    assert any("uvh model" in p for p in r.problems)


def test_bad_json_config_flagged(tmp_path):
    v = tmp_path / "x.mp4"
    v.write_bytes(b"x" * (preflight.MIN_VIDEO_BYTES + 1))
    cfg = tmp_path / "cfg.json"
    cfg.write_text("not json {{")
    r = preflight.run_checks(
        video=str(v),
        bev_config=str(cfg),
        out_csv=str(tmp_path / "out.csv"),
    )
    assert not r.ok
    assert any("bev config" in p for p in r.problems)


def test_valid_config_passes(tmp_path):
    v = tmp_path / "x.mp4"
    v.write_bytes(b"x" * (preflight.MIN_VIDEO_BYTES + 1))
    cfg = tmp_path / "cfg.json"
    cfg.write_text('{"x_min": 0}')
    r = preflight.run_checks(
        video=str(v),
        bev_config=str(cfg),
        out_csv=str(tmp_path / "out.csv"),
    )
    checks = [c for c in r.checks if c["check"] == "bev config"]
    assert checks and checks[0]["ok"]


def test_report_to_dict_shape():
    r = preflight.PreflightReport()
    r.add("test", True, "ok")
    d = r.to_dict()
    assert set(d.keys()) == {"ok", "checks", "problems"}
    assert d["ok"] is True
