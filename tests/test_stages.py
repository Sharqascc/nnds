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
    """The UVH-COCO pipeline emits stages 1..N, N = len(CANONICAL_STAGES).

    AST-based, not substring: ``ruff format`` may rewrap a long
    ``stages.stage(N, ...)`` call across multiple lines, breaking a naive
    ``f"stages.stage({n}," in src`` check (this is exactly what happened in
    commit 4b5e55f — the call for stage 5 was reformatted multi-line).
    ``tests/test_stages_ordered.py`` covers the same invariant from a
    different angle; both files exercise it independently.
    """
    import ast
    from pathlib import Path

    from src.analysis._stages import CANONICAL_STAGES

    src = Path("src/analysis/grid_trajectory/uvh_coco_fused_grid_pet.py").read_text()
    tree = ast.parse(src)
    total = len(CANONICAL_STAGES)

    stage_nums: list[int] = []
    begin_total: int | None = None
    has_done = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not (
            isinstance(f, ast.Attribute)
            and isinstance(f.value, ast.Name)
            and f.value.id == "stages"
        ):
            continue
        if f.attr == "stage" and node.args and isinstance(node.args[0], ast.Constant):
            v = node.args[0].value
            if isinstance(v, int):
                stage_nums.append(v)
        elif f.attr == "begin" and node.args and isinstance(node.args[0], ast.Constant):
            v = node.args[0].value
            if isinstance(v, int):
                begin_total = v
        elif f.attr == "done":
            has_done = True

    stage_nums.sort()
    assert stage_nums == list(range(1, total + 1)), (
        f"pipeline stages should be 1..{total}, got {stage_nums}"
    )
    assert begin_total == total, f"begin({begin_total}) should equal len(CANONICAL_STAGES)={total}"
    assert has_done, "stages.done(...) missing"
