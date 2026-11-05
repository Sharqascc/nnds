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


def test_yolo_cpu_checks_explicit_weights(tmp_path):
    """A nonexistent explicit --yolo-weights must fail the yolo-cpu report."""
    missing = tmp_path / "no-such-weights.pt"
    report = preflight.run_checks(detector="yolo-cpu", yolo_weights=str(missing))
    assert not report.ok
    assert any("yolo weights" in p for p in report.problems), report.problems


def test_yolo_cpu_checks_default_weights_when_arg_omitted(monkeypatch):
    """`--detector yolo-cpu` with no --yolo-weights must check data/models/yolo11n.pt."""
    seen_paths = []
    original = preflight._check_file

    def spy(r, path, name, min_bytes, fix):
        seen_paths.append((name, path))
        original(r, path, name, min_bytes, fix)

    monkeypatch.setattr(preflight, "_check_file", spy)
    preflight.run_checks(detector="yolo-cpu")
    yolo_calls = [p for name, p in seen_paths if "yolo" in name.lower()]
    assert yolo_calls, "yolo-cpu did not check any weights"
    assert "data/models/yolo11n.pt" in yolo_calls[0]


def test_yolo_cpu_valid_weights_passes(tmp_path):
    """A real weights file makes the yolo-cpu check pass."""
    weights = tmp_path / "yolo11n.pt"
    weights.write_bytes(b"x" * (preflight.MIN_WEIGHT_BYTES + 1))
    report = preflight.run_checks(detector="yolo-cpu", yolo_weights=str(weights))
    yolo_checks = [c for c in report.checks if "yolo weights" in c["check"]]
    assert yolo_checks and yolo_checks[0]["ok"], report.problems


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
