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


def test_pipeline_file_calls_all_canonical_stages():
    """The UVH-COCO pipeline should emit every stage in CANONICAL_STAGES.

    Count-agnostic by design: if CANONICAL_STAGES grows or shrinks, this
    test follows. The number 6 is not hardcoded here — see
    ``src/analysis/_stages.CANONICAL_STAGES`` for the source of truth.
    """
    from pathlib import Path

    from src.analysis._stages import CANONICAL_STAGES

    src = Path("src/analysis/grid_trajectory/uvh_coco_fused_grid_pet.py").read_text()
    total = len(CANONICAL_STAGES)
    for n in range(1, total + 1):
        assert f"stages.stage({n}," in src, (
            f"stage {n}/{total} missing from pipeline file; "
            f"CANONICAL_STAGES has {total} entries"
        )
    assert f"stages.begin({total}," in src, (
        f"begin() total does not match len(CANONICAL_STAGES)={total}"
    )
    assert "stages.done(" in src
