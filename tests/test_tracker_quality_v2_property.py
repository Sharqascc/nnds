"""Property tests for paper.analysis.tracker_quality_v2."""

from __future__ import annotations

import pytest

from paper.analysis import tracker_quality_v2 as tqv2


def test_quality_is_in_unit_interval():
    m = {"length": 100, "gap_rate": 0.0, "heading_jitter_rate": 0.0}
    q = tqv2._quality(m)
    assert 0.0 <= q <= 1.0


def test_quality_low_on_degenerate_track():
    """A track with minimal length and maximal gap/jitter gets a low score."""
    m = {"length": 1, "gap_rate": 1.0, "heading_jitter_rate": 1.0}
    q = tqv2._quality(m)
    # Formula: 0.4 * min(1, length/60) + 0.4 * (1 - gap) + 0.2 * (1 - jitter)
    # = 0.4 * (1/60) + 0 + 0 = 0.00667
    assert q < 0.05


def test_quality_monotone_in_length():
    m1 = {"length": 10, "gap_rate": 0.0, "heading_jitter_rate": 0.0}
    m2 = {"length": 100, "gap_rate": 0.0, "heading_jitter_rate": 0.0}
    assert tqv2._quality(m1) <= tqv2._quality(m2)


def test_metrics_on_empty_track():
    m = tqv2._metrics({})
    assert m["length"] == 0
    assert m["gap_rate"] == 0.0
    assert m["gap_bucket"] == "n/a"
