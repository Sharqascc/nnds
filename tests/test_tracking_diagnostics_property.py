"""Property tests for paper.analysis.tracking_diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from paper.analysis import tracking_diagnostics as td


def test_to_cell_px_returns_int_tuple():
    c = td._to_cell_px(0.0, 0.0)
    assert isinstance(c, tuple)
    assert len(c) == 2
    assert all(isinstance(x, int) for x in c)


def test_to_cell_px_monotone_in_x():
    a = td._to_cell_px(0.0, 0.0)
    b = td._to_cell_px(100.0, 0.0)
    assert b[0] >= a[0]


def test_track_lengths_shape_on_simple_track():
    tracks = {1: {0: (0.0, 0.0, 0.0, 0.0), 1: (0.0, 0.0, 1.0, 0.0), 2: (0.0, 0.0, 2.0, 0.0)}}
    out = td._track_lengths(tracks)
    assert out["n_tracks"] == 1
    assert out["min"] == 3
    assert out["max"] == 3


def test_gap_rate_no_gaps_is_zero():
    tracks = {1: {0: (0.0, 0.0, 0.0, 0.0), 1: (0.0, 0.0, 0.0, 0.0), 2: (0.0, 0.0, 0.0, 0.0)}}
    out = td._gap_rate(tracks)
    assert out["mean"] == pytest.approx(0.0)


def test_speed_plausibility_flags_fast_track():
    tracks = {
        1: {0: (0.0, 0.0, 0.0, 0.0), 30: (0.0, 0.0, 500.0, 0.0)},  # 500 m in 1 s
    }
    out = td._speed_plausibility(tracks)
    assert out["implausible_count"] >= 1
