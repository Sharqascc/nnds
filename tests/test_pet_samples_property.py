"""Property-based tests for src.analysis.visualization.pet_samples.

Each test is marked @pytest.mark.property and runs in the CI's
"Property tests (hypothesis ci-pr)" lane on every PR.

These complement the example-based tests in test_pet_samples.py:
examples catch specific known cases; properties exhaustively verify
universal invariants across all possible inputs.

The primary subject here is `_branch_frames`, which decides which two
frame numbers appear on the PET sample panel and in the ring markers.
A regression in this function is exactly what caused the "wrong label"
bug reported against PR #84's output.
"""

from __future__ import annotations

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from src.analysis.visualization.pet_samples import _branch_frames

# ── Strategies ─────────────────────────────────────────────────────────

# Non-negative integer frame numbers up to 10000 (~5 min at 30 fps).
_FRAME = st.integers(min_value=0, max_value=10000)
_FRAME_OR_NONE = st.one_of(_FRAME, st.none())


# ── Invariant 1: labels come from a fixed pair set ─────────────────────


@pytest.mark.property
@given(a_entry=_FRAME_OR_NONE, a_exit=_FRAME_OR_NONE, b_entry=_FRAME_OR_NONE, b_exit=_FRAME_OR_NONE)
def test_labels_are_valid_pair(a_entry, a_exit, b_entry, b_exit):
    """Labels must always be one of the two defined (A, B) pairs."""
    _, _, a_label, b_label = _branch_frames(a_entry, a_exit, b_entry, b_exit)
    assert (a_label, b_label) in {
        ("A out", "B in"),
        ("A in", "B out"),
    }


# ── Invariant 2: case-1 exactness (A_exit < B_entry) ───────────────────


@pytest.mark.property
@given(a_entry=_FRAME, a_exit=_FRAME, b_entry=_FRAME, b_exit=_FRAME)
def test_case1_exact(a_entry, a_exit, b_entry, b_exit):
    """When a_exit < b_entry, result MUST be (a_exit, b_entry, A out, B in)."""
    assume(a_exit < b_entry)
    result = _branch_frames(a_entry, a_exit, b_entry, b_exit)
    assert result == (a_exit, b_entry, "A out", "B in")


# ── Invariant 3: case-2 exactness (B_exit < A_entry, case 1 not met) ───


@pytest.mark.property
@given(a_entry=_FRAME, a_exit=_FRAME, b_entry=_FRAME, b_exit=_FRAME)
def test_case2_exact(a_entry, a_exit, b_entry, b_exit):
    """When case-1 does not fire but b_exit < a_entry, use A-in / B-out."""
    assume(a_exit >= b_entry)
    assume(b_exit < a_entry)
    result = _branch_frames(a_entry, a_exit, b_entry, b_exit)
    assert result == (a_entry, b_exit, "A in", "B out")


# ── Invariant 4: case-1 priority when both are technically possible ────


@pytest.mark.property
@given(a_entry=_FRAME, a_exit=_FRAME, b_entry=_FRAME, b_exit=_FRAME)
def test_case1_has_priority(a_entry, a_exit, b_entry, b_exit):
    """When both branch conditions are true, case 1 wins (it is checked first)."""
    assume(a_exit < b_entry)
    assume(b_exit < a_entry)
    _, _, a_label, b_label = _branch_frames(a_entry, a_exit, b_entry, b_exit)
    assert (a_label, b_label) == ("A out", "B in")


# ── Invariant 5: label-frame internal consistency ──────────────────────


@pytest.mark.property
@given(a_entry=_FRAME_OR_NONE, a_exit=_FRAME_OR_NONE, b_entry=_FRAME_OR_NONE, b_exit=_FRAME_OR_NONE)
def test_label_frame_consistency(a_entry, a_exit, b_entry, b_exit):
    """The returned frame must be the one named by the returned label."""
    a_frame, b_frame, a_label, b_label = _branch_frames(a_entry, a_exit, b_entry, b_exit)
    expected_a = a_exit if a_label == "A out" else a_entry
    expected_b = b_entry if b_label == "B in" else b_exit
    assert a_frame == expected_a
    assert b_frame == expected_b


# ── Invariant 6: returned frames are subsets of the inputs ─────────────


@pytest.mark.property
@given(a_entry=_FRAME_OR_NONE, a_exit=_FRAME_OR_NONE, b_entry=_FRAME_OR_NONE, b_exit=_FRAME_OR_NONE)
def test_frames_subset_of_inputs(a_entry, a_exit, b_entry, b_exit):
    """The function never invents frames — every output is a subset of input."""
    a_frame, b_frame, _, _ = _branch_frames(a_entry, a_exit, b_entry, b_exit)
    assert a_frame in (a_entry, a_exit)
    assert b_frame in (b_entry, b_exit)


# ── Invariant 7: None preservation ─────────────────────────────────────


@pytest.mark.property
@given(a_entry=_FRAME_OR_NONE, a_exit=_FRAME_OR_NONE, b_entry=_FRAME_OR_NONE, b_exit=_FRAME_OR_NONE)
def test_none_preservation(a_entry, a_exit, b_entry, b_exit):
    """A returned frame is None only if its source input is None."""
    a_frame, b_frame, _, _ = _branch_frames(a_entry, a_exit, b_entry, b_exit)
    if a_frame is None:
        assert a_exit is None
    if b_frame is None:
        assert b_entry is None
