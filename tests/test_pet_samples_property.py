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


# ═══════════════════════════════════════════════════════════════════════
# _frame_point invariants
# ═══════════════════════════════════════════════════════════════════════
#
# _frame_point(traj, frame_idx) returns the (x, y) coords of the first
# point in `traj` whose int(truncated) frame equals int(frame_idx), or
# None if no such point exists. It is used to place the A/B ring markers
# on the sample image at the frame the branch logic selected.

_COORD = st.floats(min_value=-10000, max_value=10000, allow_nan=False, allow_infinity=False)
_FLOAT_FRAME = st.floats(min_value=0, max_value=10000, allow_nan=False, allow_infinity=False)
_TRAJ_PT_INT = st.tuples(_COORD, _COORD, _FRAME)
_TRAJ_PT_FLOAT = st.tuples(_COORD, _COORD, _FLOAT_FRAME)
_TRAJ_INT = st.lists(_TRAJ_PT_INT, min_size=0, max_size=50)
_TRAJ_FLOAT = st.lists(_TRAJ_PT_FLOAT, min_size=0, max_size=50)


@pytest.mark.property
@given(traj=_TRAJ_INT, frame=_FRAME_OR_NONE)
def test_hit_iff_present(traj, frame):
    """_frame_point returns non-None iff some point matches the frame."""
    from src.analysis.visualization.pet_samples import _frame_point

    if frame is None:
        assert _frame_point(traj, frame) is None
        return
    has_match = any(int(p[2]) == int(frame) for p in traj)
    result = _frame_point(traj, frame)
    assert (result is not None) == has_match, (
        f"traj={traj}, frame={frame}, has_match={has_match}, result={result}"
    )


@pytest.mark.property
@given(traj=_TRAJ_INT, frame=_FRAME)
def test_result_matches_first_matching_point(traj, frame):
    """When multiple points share a frame, the FIRST one wins."""
    from src.analysis.visualization.pet_samples import _frame_point

    matches = [p for p in traj if int(p[2]) == frame]
    result = _frame_point(traj, frame)
    if not matches:
        assert result is None
    else:
        first = matches[0]
        assert result == (int(first[0]), int(first[1])), (
            f"traj={traj}, frame={frame}, result={result}, "
            f"expected first match {(int(first[0]), int(first[1]))}"
        )


@pytest.mark.property
@given(traj=_TRAJ_INT)
def test_none_frame_returns_none(traj):
    """frame_idx=None always returns None regardless of trajectory."""
    from src.analysis.visualization.pet_samples import _frame_point

    assert _frame_point(traj, None) is None


@pytest.mark.property
@given(frame=_FRAME_OR_NONE)
def test_empty_traj_returns_none(frame):
    """Empty trajectory always returns None regardless of frame."""
    from src.analysis.visualization.pet_samples import _frame_point

    assert _frame_point([], frame) is None


@pytest.mark.property
@given(traj=_TRAJ_FLOAT, frame=_FRAME)
def test_float_frames_truncated_for_comparison(traj, frame):
    """Float frames are compared as int(p[2]) == int(frame_idx)."""
    from src.analysis.visualization.pet_samples import _frame_point

    # Query a frame that may match some truncated float
    result = _frame_point(traj, frame)
    # Whether None or non-None, verify the invariant: result is
    # consistent with the truncation-based matching rule
    has_match = any(int(p[2]) == int(frame) for p in traj)
    assert (result is not None) == has_match


@pytest.mark.property
@given(
    traj=st.lists(_TRAJ_PT_INT, min_size=1, max_size=20),
    frame_offset=st.integers(min_value=0, max_value=10000),
)
def test_returned_coords_are_truncated_ints(traj, frame_offset):
    """If not None, returned coords are int(x), int(y) of some trajectory point."""
    from src.analysis.visualization.pet_samples import _frame_point

    # Pick a frame that certainly exists in traj
    target_frame = int(traj[0][2])
    result = _frame_point(traj, target_frame)
    assert result is not None
    # Must be (int(x), int(y)) for some matching point
    expected_coords = {(int(p[0]), int(p[1])) for p in traj if int(p[2]) == target_frame}
    assert result in expected_coords
    # Must be a tuple of ints (not floats, not numpy)
    assert isinstance(result[0], int)
    assert isinstance(result[1], int)
