"""Property tests for src/analysis/visualization/class_colors.py.

Complements tests/test_class_colors.py (unit tests) by asserting invariants
that should hold for ANY input — real class names, fake class names, spaces,
mixed case, arbitrary track ids.

Property focus:
  * color_for / label_for never crash and always return valid types.
  * Known classes: case-insensitive lookup, whitespace stripped, palette-bound.
  * Fallback behaviour: unknown class + track_id picks from _FALLBACK_PALETTE.
  * No-class case returns FALLBACK_DEFAULT.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.analysis.visualization.class_colors import (
    _FALLBACK_PALETTE,
    CLASS_COLORS,
    FALLBACK_DEFAULT,
    OKABE_ITO_BGR,
    color_for,
    label_for,
)


NAME = st.text(
    min_size=0,
    max_size=30,
    alphabet=st.characters(min_codepoint=32, max_codepoint=126),
)
NON_WS_NAME = NAME.filter(lambda s: s.strip())
TRACK_ID = st.integers(min_value=-10, max_value=1_000_000)
TRACK_ID_POS = st.integers(min_value=0, max_value=10_000)
VALID_CLASS = st.sampled_from(sorted(CLASS_COLORS))
EMPTY_ISH = st.one_of(
    st.just(""),
    st.text(alphabet=" \t\n\r", min_size=1, max_size=10),
)


@pytest.mark.property
@given(name=NAME, track_id=TRACK_ID)
@settings(max_examples=50, deadline=None)
def test_color_for_returns_valid_bgr(name, track_id):
    color = color_for(name, track_id)
    assert isinstance(color, tuple)
    assert len(color) == 3
    assert all(isinstance(c, int) and 0 <= c <= 255 for c in color)


@pytest.mark.property
@given(name=NAME, track_id=TRACK_ID)
@settings(max_examples=30, deadline=None)
def test_color_for_is_idempotent(name, track_id):
    assert color_for(name, track_id) == color_for(name, track_id)


@pytest.mark.property
@given(name=VALID_CLASS)
@settings(max_examples=50, deadline=None)
def test_color_for_known_class_returns_stored_colour(name):
    assert color_for(name) == CLASS_COLORS[name]


@pytest.mark.property
@given(name=VALID_CLASS)
@settings(max_examples=50, deadline=None)
def test_color_for_is_case_insensitive(name):
    original = color_for(name)
    assert color_for(name.lower()) == original
    assert color_for(name.upper()) == original


@pytest.mark.property
@given(name=VALID_CLASS)
@settings(max_examples=50, deadline=None)
def test_color_for_strips_whitespace(name):
    assert color_for(f"  {name}  ") == color_for(name)


@pytest.mark.property
@given(name=VALID_CLASS)
@settings(max_examples=50, deadline=None)
def test_color_for_class_colour_is_from_okabe_ito(name):
    palette_values = set(OKABE_ITO_BGR.values())
    assert color_for(name) in palette_values


@pytest.mark.property
@given(track_id=TRACK_ID_POS)
@settings(max_examples=50, deadline=None)
def test_color_for_unknown_class_uses_track_fallback(track_id):
    color = color_for("__no_such_class_xyz__", track_id)
    assert color in _FALLBACK_PALETTE


@pytest.mark.property
@given(name=EMPTY_ISH)
@settings(max_examples=20, deadline=None)
def test_color_for_none_or_empty_returns_default(name):
    assert color_for(name) == FALLBACK_DEFAULT


@pytest.mark.property
@given(name=NAME, track_id=TRACK_ID)
@settings(max_examples=30, deadline=None)
def test_label_for_returns_non_empty_string(name, track_id):
    label = label_for(name, track_id)
    assert isinstance(label, str)
    assert len(label) > 0


@pytest.mark.property
@given(name=NON_WS_NAME, track_id=TRACK_ID_POS)
@settings(max_examples=30, deadline=None)
def test_label_for_includes_track_id_when_valid(name, track_id):
    assert f"#{track_id}" in label_for(name, track_id)


@pytest.mark.property
@given(name=NON_WS_NAME)
@settings(max_examples=30, deadline=None)
def test_label_for_without_track_has_no_hash(name):
    label = label_for(name, None)
    assert not label.startswith("#")


def test_label_for_none_none_is_question_mark():
    """Explicit boundary — the documented return for both-None."""
    assert label_for(None, None) == "?"
