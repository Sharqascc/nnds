"""Tests for stages.progress() — the granular progress helper."""

from __future__ import annotations

import io
from contextlib import redirect_stdout

from src.analysis import _stages as stages


def test_progress_noop_when_quiet():
    buf = io.StringIO()
    with redirect_stdout(buf):
        stages.begin(3, header="t", quiet=True)
        stages.stage(2, "Detect")
        for i in range(1, 301):
            stages.progress(i, 300, "frames")
    out = buf.getvalue()
    assert "[progress" not in out
    assert "[stage 2/" not in out


def test_progress_prints_at_boundaries():
    buf = io.StringIO()
    with redirect_stdout(buf):
        stages.begin(3, header="t")
        stages.stage(2, "Detect")
        for i in range(1, 301):
            stages.progress(i, 300, "frames", every=100)
    out = buf.getvalue()
    assert "100/300" in out
    assert "200/300" in out
    assert "300/300" in out
    assert "[stage 2/3]" in out


def test_progress_unknown_total_prints_at_every():
    buf = io.StringIO()
    with redirect_stdout(buf):
        stages.begin(3, header="t")
        stages.stage(4, "render")
        for i in range(1, 251):
            stages.progress(i, 0, "render", every=100)
    out = buf.getvalue()
    assert "100/?" in out
    assert "200/?" in out


def test_progress_stage_num_zero_uses_progress_prefix():
    buf = io.StringIO()
    with redirect_stdout(buf):
        stages.begin(3, header="t")
        for i in range(1, 101):
            stages.progress(i, 100, "render", every=100, stage_num=0)
    out = buf.getvalue()
    assert "[progress]" in out
    assert "100/100" in out


def test_stage_updates_current_stage():
    stages.begin(3, header="t", quiet=True)  # quiet but stage still tracks
    stages.stage(2, "Detect")
    assert stages._CURRENT_STAGE == 2


def test_begin_resets_current_stage():
    stages.begin(3, header="t", quiet=True)
    stages.stage(5, "x")
    stages.begin(5, header="t2", quiet=True)
    assert stages._CURRENT_STAGE == 0


def test_progress_called_before_any_stage():
    """progress() before any stage() should use [progress] prefix."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        stages.begin(3, header="t")
        for i in range(1, 101):
            stages.progress(i, 100, "early", every=100)
    assert "[progress]" in buf.getvalue()
