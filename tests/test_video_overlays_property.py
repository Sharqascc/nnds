"""Property tests for pure helpers in video_overlays.VideoOverlayPlotter."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.analysis.visualization import video_overlays as vo


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
    assert label in {"CRITICAL", "SERIOUS", "MODERATE", "SAFE", "LOW"}


def test_severity_monotone_in_pet():
    """Monotone in the sense that severity classes are ordered by PET."""
    p = vo.VideoOverlayPlotter()
    # Boundaries: critical < 0.5, serious < 1.0, moderate < 1.5, safe < 5.0
    lc = p._get_severity_label(0.1)
    ls = p._get_severity_label(0.7)
    lm = p._get_severity_label(1.2)
    lsafe = p._get_severity_label(4.0)
    assert lc == "CRITICAL"
    assert ls == "SERIOUS"
    assert lm == "MODERATE"
    assert lsafe == "SAFE"


def test_default_thresholds_are_ordered():
    t = vo.DEFAULT_THRESHOLDS
    assert t["critical"] < t["serious"] < t["moderate"] < t["safe"]


def test_colors_bgr_has_required_keys():
    expected = {"blue", "orange", "green", "yellow", "purple", "cyan", "red", "black"}
    assert expected.issubset(vo.COLORS_BGR.keys())
    for name, val in vo.COLORS_BGR.items():
        assert isinstance(val, tuple)
        assert len(val) == 3


def test_constructor_accepts_custom_thresholds():
    p = vo.VideoOverlayPlotter(thresholds={"critical": 0.2, "serious": 0.5,
                                            "moderate": 1.0, "safe": 3.0})
    assert p._get_severity_label(0.1) == "CRITICAL"
    assert p._get_severity_label(2.0) == "SAFE"
