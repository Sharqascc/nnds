"""Tests for the two preflight fixes: ffmpeg optional, yolo-cpu checks weights."""

from __future__ import annotations

import pytest

from src.pipeline import preflight


def test_ffmpeg_missing_is_warning_not_failure(monkeypatch):
    """No ffmpeg binary should not fail the report."""
    monkeypatch.setattr(
        "shutil.which",
        lambda name: None if name == "ffmpeg" else "/usr/bin/x",
    )
    r = preflight.PreflightReport()
    preflight._check_ffmpeg(r)

    assert r.problems == [], f"ffmpeg absence leaked into problems: {r.problems}"
    assert any(c["check"] == "ffmpeg on PATH" and c["ok"] for c in r.checks)
    note = next(c for c in r.checks if c["check"] == "ffmpeg on PATH")
    assert "optional" in note["detail"].lower()


def test_ffmpeg_present_passes(monkeypatch):
    monkeypatch.setattr(
        "shutil.which",
        lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else "/usr/bin/x",
    )
    r = preflight.PreflightReport()
    preflight._check_ffmpeg(r)
    assert r.problems == []
    note = next(c for c in r.checks if c["check"] == "ffmpeg on PATH")
    assert note["ok"]
    assert note["detail"] == "/usr/bin/ffmpeg"


def test_uvh_coco_fused_does_not_require_yolo_weights(tmp_path):
    """uvh-coco-fused should not require --yolo-weights; coco_person_model is the check."""
    coco = tmp_path / "yolo11n.pt"
    coco.write_bytes(b"x" * (preflight.MIN_WEIGHT_BYTES + 1))
    uvh = tmp_path / "uvh26.pt"
    uvh.write_bytes(b"x" * (preflight.MIN_WEIGHT_BYTES + 1))

    report = preflight.run_checks(
        detector="uvh-coco-fused",
        uvh_model=str(uvh),
        coco_person_model=str(coco),
    )
    weight_problems = [p for p in report.problems if "weights" in p or "model" in p]
    assert not weight_problems, f"unexpected weight problems: {weight_problems}"
