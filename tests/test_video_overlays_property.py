"""Property tests for pure helpers in video_overlays.VideoOverlayPlotter."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.analysis.visualization import video_overlays as vo


# The module implements five severity levels (verified against source):
#   CRITICAL < critical
#   SERIOUS  < serious
#   MODERATE < moderate
#   SLIGHT   < safe
#   SAFE     (else)
KNOWN_LABELS = {"CRITICAL", "SERIOUS", "MODERATE", "SLIGHT", "SAFE"}


@given(pet=st.floats(0.0, 10.0, allow_nan=False, allow_infinity=False))
def test_severity_color_returns_bgr_tuple(pet):
    p = vo.VideoOverlayPlotter()
    c = p._get_severity_color(pet)
    assert isinstance(c, tuple)
    assert len(c) == 3
    assert all(isinstance(x, int) and 0 <= x <= 255 for x in c)


@given(pet=st.floats(0.0, 10.0, allow_nan=False, allow_infinity=False))
def test_severity_label_is_known_string(pet):
    p = vo.VideoOverlayPlotter()
    label = p._get_severity_label(pet)
    assert isinstance(label, str)
    assert label in KNOWN_LABELS


def test_severity_labels_at_default_thresholds():
    """Boundary values at default thresholds produce the correct label."""
    p = vo.VideoOverlayPlotter()
    # Default: critical=0.5, serious=1.0, moderate=1.5, safe=5.0
    assert p._get_severity_label(0.1) == "CRITICAL"
    assert p._get_severity_label(0.7) == "SERIOUS"
    assert p._get_severity_label(1.2) == "MODERATE"
    assert p._get_severity_label(2.0) == "SLIGHT"
    assert p._get_severity_label(7.0) == "SAFE"


def test_default_thresholds_are_ordered():
    t = vo.DEFAULT_THRESHOLDS
    assert t["critical"] < t["serious"] < t["moderate"] < t["safe"]


def test_colors_bgr_has_required_keys():
    expected = {"blue", "orange", "green", "yellow", "purple", "cyan", "red", "black"}
    assert expected.issubset(vo.COLORS_BGR.keys())
    for name, val in vo.COLORS_BGR.items():
        assert isinstance(val, tuple)
        assert len(val) == 3
        assert all(isinstance(x, int) and 0 <= x <= 255 for x in val)


def test_constructor_accepts_custom_thresholds():
    """Custom thresholds change the label boundaries accordingly."""
    p = vo.VideoOverlayPlotter(
        thresholds={"critical": 0.2, "serious": 0.5,
                    "moderate": 1.0, "safe": 3.0}
    )
    assert p._get_severity_label(0.1) == "CRITICAL"
    assert p._get_severity_label(0.3) == "SERIOUS"
    assert p._get_severity_label(0.7) == "MODERATE"
    assert p._get_severity_label(2.0) == "SLIGHT"
    assert p._get_severity_label(5.0) == "SAFE"


@given(
    critical=st.floats(0.01, 0.5, allow_nan=False, allow_infinity=False),
    serious=st.floats(0.51, 1.0, allow_nan=False, allow_infinity=False),
    moderate=st.floats(1.01, 2.0, allow_nan=False, allow_infinity=False),
    safe=st.floats(2.01, 10.0, allow_nan=False, allow_infinity=False),
)
def test_label_monotone_in_ordering(critical, serious, moderate, safe):
    """A smaller PET always maps to the same or an earlier severity."""
    p = vo.VideoOverlayPlotter(
        thresholds={"critical": critical, "serious": serious,
                    "moderate": moderate, "safe": safe}
    )
    ordered = ["CRITICAL", "SERIOUS", "MODERATE", "SLIGHT", "SAFE"]
    ranks = []
    for pet in [0.0, critical, serious, moderate, safe, safe + 1.0]:
        label = p._get_severity_label(pet)
        ranks.append(ordered.index(label))
    # Non-decreasing rank as pet increases
    for i in range(1, len(ranks)):
        assert ranks[i] >= ranks[i - 1]
