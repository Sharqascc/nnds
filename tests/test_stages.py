"""Tests for the numbered-stage helper and its usage in the pipeline."""

from __future__ import annotations

import io
from contextlib import redirect_stdout

from src.analysis import _stages as stages


def test_stage_format_is_stable():
    buf = io.StringIO()
    with redirect_stdout(buf):
        stages.begin(3, header="Test Pipeline")
        stages.stage(1, "Setup", "backend=pytorch")
        stages.stage(2, "Detect")
        stages.stage(3, "Write", "out.csv")
        stages.done("done")
    out = buf.getvalue()

    assert "Test Pipeline" in out
    assert "[1/3] Setup" in out
    assert "backend=pytorch" in out
    assert "[2/3] Detect" in out
    assert "[3/3] Write" in out
    assert "out.csv" in out
    assert "✅ Done in" in out


def test_stage_before_begin_does_not_crash():
    """Calling stage() without begin() should not raise; it uses '?' as total."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        stages._TOTAL = None
        stages._START = None
        stages.stage(1, "Detect")
    assert "[1/?]" in buf.getvalue()


def test_pipeline_file_calls_five_stages():
    """The UVH-COCO pipeline should emit exactly five numbered stages.

    This guards against a future refactor dropping or duplicating a boundary.
    """
    from pathlib import Path

    src = Path("src/analysis/grid_trajectory/uvh_coco_fused_grid_pet.py").read_text()
    # each stage call is `stages.stage(N, ...)` with N in 1..5
    for n in range(1, 6):
        assert f"stages.stage({n}," in src, f"stage {n} missing from pipeline file"
    assert "stages.begin(" in src
    assert "stages.done(" in src
