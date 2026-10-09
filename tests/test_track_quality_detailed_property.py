"""Property tests for paper.analysis.track_quality_detailed."""

from __future__ import annotations

import pytest

from paper.analysis import track_quality_detailed as tqd

KNOWN_FLAGS = {"short", "gappy", "jumpy", "fast", "huge_jump"}


def test_flags_are_subset_of_known():
    m = {
        "length": 100,
        "gap_rate": 0.0,
        "heading_p95_deg": 10.0,
        "max_speed_mps": 5.0,
        "max_jump_m": 1.0,
    }
    assert set(tqd._flags(m)).issubset(KNOWN_FLAGS)


def test_short_flag_triggers_on_short_track():
    m = {
        "length": 5,
        "gap_rate": 0.0,
        "heading_p95_deg": 0.0,
        "max_speed_mps": 0.0,
        "max_jump_m": 0.0,
    }
    assert "short" in tqd._flags(m)


def test_gappy_flag_triggers_on_high_gap_rate():
    m = {
        "length": 100,
        "gap_rate": 0.9,
        "heading_p95_deg": 0.0,
        "max_speed_mps": 0.0,
        "max_jump_m": 0.0,
    }
    assert "gappy" in tqd._flags(m)


def test_track_metrics_on_static_track():
    td = {0: (1.0, 1.0, 1.0, 1.0),
          1: (1.0, 1.0, 1.0, 1.0),
          2: (1.0, 1.0, 1.0, 1.0)}
    m = tqd._track_metrics(td)
    assert m["length"] == 3
    assert m["span"] == 3
    assert m["gap_rate"] == pytest.approx(0.0)
    assert m["max_gap_frames"] == 0
