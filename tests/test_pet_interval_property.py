"""Property-based tests for pet_interval.compute_pet_from_intervals."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pet_interval import compute_pet_from_frames, compute_pet_from_intervals

_FLOAT = st.floats(
    min_value=-1_000.0,
    max_value=1_000.0,
    allow_nan=False,
    allow_infinity=False,
)


@st.composite
def valid_interval(draw):
    start = draw(_FLOAT)
    length = draw(
        st.floats(min_value=0.0, max_value=1_000.0, allow_nan=False, allow_infinity=False)
    )
    return start, start + length


@given(a=valid_interval(), b=valid_interval())
def test_pet_non_negative_or_none(a, b):
    result = compute_pet_from_intervals(*a, *b)
    if result.pet_s is not None:
        assert result.pet_s >= 0.0


@given(a=valid_interval(), b=valid_interval())
def test_status_consistent_with_pet_value(a, b):
    result = compute_pet_from_intervals(*a, *b)
    if result.pet_status == "sequential":
        assert result.pet_s is not None
        assert result.first_actor in ("a", "b")
        assert result.second_actor in ("a", "b")
        assert result.first_actor != result.second_actor
        assert result.overlap_duration_s == 0.0
    elif result.pet_status == "overlap":
        assert result.pet_s is None
        assert result.first_actor is None
        assert result.second_actor is None
        assert result.overlap_duration_s > 0.0


@given(a=valid_interval(), b=valid_interval())
def test_sequential_pet_equals_exact_gap(a, b):
    result = compute_pet_from_intervals(*a, *b)
    if result.pet_status != "sequential":
        return
    if result.first_actor == "a":
        assert result.pet_s == pytest.approx(b[0] - a[1])
    else:
        assert result.pet_s == pytest.approx(a[0] - b[1])


@given(a=valid_interval(), b=valid_interval())
def test_swap_symmetry(a, b):
    r1 = compute_pet_from_intervals(*a, *b)
    r2 = compute_pet_from_intervals(*b, *a)
    if r1.pet_s is None:
        assert r2.pet_s is None
        assert r1.pet_status == r2.pet_status == "overlap"
    else:
        assert r2.pet_s == pytest.approx(r1.pet_s)
        # First/second actors must swap when arguments swap
        assert r1.first_actor != r2.first_actor


@given(a=valid_interval(), b=valid_interval())
def test_overlap_duration_bounded(a, b):
    result = compute_pet_from_intervals(*a, *b)
    assert result.overlap_duration_s >= 0.0
    if result.pet_status == "overlap":
        a_len = a[1] - a[0]
        b_len = b[1] - b[0]
        assert result.overlap_duration_s <= min(a_len, b_len) + 1e-9


@given(a=valid_interval(), b=valid_interval())
def test_invalid_entry_after_exit_raises(a, b):
    with pytest.raises(ValueError):
        compute_pet_from_intervals(a[1], a[0] - 1.0, b[0], b[1])
        # a_entry > a_exit triggers the check


@given(a=valid_interval())
def test_nan_raises(a):
    with pytest.raises(ValueError):
        compute_pet_from_intervals(float("nan"), a[1], a[0], a[1])


@given(a=valid_interval())
def test_infinity_raises(a):
    with pytest.raises(ValueError):
        compute_pet_from_intervals(a[0], float("inf"), a[0], a[1])


@given(
    a_entry=st.integers(-1000, 1000),
    a_dur=st.integers(0, 1000),
    b_entry=st.integers(-1000, 1000),
    b_dur=st.integers(0, 1000),
    fps=st.floats(min_value=1.0, max_value=200.0, allow_nan=False, allow_infinity=False),
)
def test_frames_matches_seconds(a_entry, a_dur, b_entry, b_dur, fps):
    a_exit = a_entry + a_dur
    b_exit = b_entry + b_dur
    r_frames = compute_pet_from_frames(a_entry, a_exit, b_entry, b_exit, fps)
    r_seconds = compute_pet_from_intervals(a_entry / fps, a_exit / fps, b_entry / fps, b_exit / fps)
    assert r_frames.pet_status == r_seconds.pet_status
    if r_seconds.pet_s is not None:
        assert r_frames.pet_s == pytest.approx(r_seconds.pet_s, abs=1e-9)


def test_invalid_fps_raises():
    with pytest.raises(ValueError):
        compute_pet_from_frames(0, 10, 20, 30, 0.0)
    with pytest.raises(ValueError):
        compute_pet_from_frames(0, 10, 20, 30, -1.0)
