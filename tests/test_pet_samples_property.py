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


# ═══════════════════════════════════════════════════════════════════════
# _build_metadata_lines invariants
# ═══════════════════════════════════════════════════════════════════════
#
# Contract: video_source, grid_cell, conflict_type are sanitized to
# single-line strings (newlines/CRs → spaces, then .strip()). Empty
# sanitized values render as "?" for video_source (and pass through
# unchanged for the others).
#
# These tests lock:
#   - 7-element structure (catches missing-comma bug)
#   - No embedded newlines anywhere
#   - Verbatim single-line preservation
#   - Actor-specific labels with correct frame resolution
#   - All input fields actually reflected in output


# Printable non-whitespace alphabet for verbatim tests.
# Excludes: control chars (Cc), surrogates (Cs), separators (Zs/Zl/Zp),
# and combining marks (Mn/Mc) that might have edge cases in assertion.
_SINGLE_LINE_TEXT = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Pd", "Po")),
    min_size=1,
    max_size=100,
)

# Full-range text for sanitization tests (includes newlines, unicode, etc.)
_ANY_TEXT = st.text(min_size=0, max_size=100)

_A_LABEL = st.sampled_from(["A out", "A in"])
_B_LABEL = st.sampled_from(["B in", "B out"])


def _make_lines(
    video_source="v.mp4",
    frame_idx=0,
    grid_cell="C",
    conflict_type="t",
    track_a=1,
    track_b=2,
    a_mark_frame=None,
    a_mark_label="A out",
    b_mark_frame=None,
    b_mark_label="B in",
):
    from src.analysis.visualization.pet_samples import _build_metadata_lines

    return _build_metadata_lines(
        video_source=video_source,
        frame_idx=frame_idx,
        grid_cell=grid_cell,
        conflict_type=conflict_type,
        track_a=track_a,
        track_b=track_b,
        a_mark_frame=a_mark_frame,
        a_mark_label=a_mark_label,
        b_mark_frame=b_mark_frame,
        b_mark_label=b_mark_label,
    )


# ── Structural invariants ──────────────────────────────────────────────


@pytest.mark.property
@given(
    video_source=_ANY_TEXT,
    frame_idx=st.integers(min_value=0, max_value=100000),
    grid_cell=_ANY_TEXT,
    conflict_type=_ANY_TEXT,
    track_a=st.integers(min_value=0, max_value=100000),
    track_b=st.integers(min_value=0, max_value=100000),
    a_mark_frame=st.one_of(st.none(), st.integers(min_value=0, max_value=100000)),
    b_mark_frame=st.one_of(st.none(), st.integers(min_value=0, max_value=100000)),
    a_mark_label=_A_LABEL,
    b_mark_label=_B_LABEL,
)
def test_always_returns_seven_separate_strings(
    video_source,
    frame_idx,
    grid_cell,
    conflict_type,
    track_a,
    track_b,
    a_mark_frame,
    b_mark_frame,
    a_mark_label,
    b_mark_label,
):
    """Exactly 7 separate strings — this is the missing-comma guard."""
    lines = _make_lines(
        video_source=video_source,
        frame_idx=frame_idx,
        grid_cell=grid_cell,
        conflict_type=conflict_type,
        track_a=track_a,
        track_b=track_b,
        a_mark_frame=a_mark_frame,
        a_mark_label=a_mark_label,
        b_mark_frame=b_mark_frame,
        b_mark_label=b_mark_label,
    )
    assert isinstance(lines, list), f"expected list, got {type(lines).__name__}"
    assert len(lines) == 7, (
        f"expected 7 separate strings, got {len(lines)}. "
        f"Missing commas between adjacent f-strings concatenate into fewer items."
    )
    assert all(isinstance(s, str) for s in lines)


@pytest.mark.property
@given(video_source=_ANY_TEXT, grid_cell=_ANY_TEXT, conflict_type=_ANY_TEXT)
def test_no_embedded_newlines_in_any_line(video_source, grid_cell, conflict_type):
    """Even with newlines in inputs, output lines never contain newlines."""
    lines = _make_lines(
        video_source=video_source,
        grid_cell=grid_cell,
        conflict_type=conflict_type,
    )
    for i, line in enumerate(lines):
        assert "\n" not in line, f"line {i} contains newline: {line!r}"
        assert "\r" not in line, f"line {i} contains CR: {line!r}"
        # Belt-and-braces: catch any line-break category
        assert len(line.splitlines()) <= 1, f"line {i} splits into multiple lines: {line!r}"


@pytest.mark.property
@given(video_source=_ANY_TEXT)
def test_video_is_always_first_line(video_source):
    lines = _make_lines(video_source=video_source)
    assert lines[0].startswith("Video:"), f"first line: {lines[0]!r}"


# ── Verbatim preservation (single-line inputs only) ────────────────────


@pytest.mark.property
@given(video_source=_SINGLE_LINE_TEXT)
def test_verbatim_single_line_preserved(video_source):
    """Single-line, non-whitespace input survives verbatim."""
    lines = _make_lines(video_source=video_source)
    assert lines[0] == f"Video: {video_source}", (
        f"single-line input not preserved: {video_source!r} → {lines[0]!r}"
    )


# ── Empty / whitespace-only sanitization ───────────────────────────────


@pytest.mark.property
@given(video_source=st.text(alphabet=" \t\n\r", min_size=0, max_size=20))
def test_empty_or_whitespace_video_renders_question_mark(video_source):
    """Empty or whitespace-only video_source renders as 'Video: ?'."""
    lines = _make_lines(video_source=video_source)
    assert lines[0] == "Video: ?", (
        f"whitespace-only input {video_source!r} should render as '?' but got {lines[0]!r}"
    )


# ── Field reflection ───────────────────────────────────────────────────


@pytest.mark.property
@given(
    frame_idx=st.integers(min_value=0, max_value=100000),
    track_a=st.integers(min_value=0, max_value=100000),
    track_b=st.integers(min_value=0, max_value=100000),
)
def test_numeric_fields_reflected(frame_idx, track_a, track_b):
    """frame_idx, track_a, track_b appear verbatim in their lines."""
    lines = _make_lines(frame_idx=frame_idx, track_a=track_a, track_b=track_b)
    assert lines[1] == f"Frame: {frame_idx}"
    assert lines[4] == f"Pair:  T{track_a} ↔ T{track_b}"


@pytest.mark.property
@given(grid_cell=_SINGLE_LINE_TEXT, conflict_type=_SINGLE_LINE_TEXT)
def test_text_fields_reflected(grid_cell, conflict_type):
    """grid_cell, conflict_type appear verbatim (after sanitization)."""
    lines = _make_lines(grid_cell=grid_cell, conflict_type=conflict_type)
    assert lines[2] == f"Grid:  {grid_cell}"
    assert lines[3] == f"Type:  {conflict_type}"


# ── Actor-specific labels (the strongest guard against PR #86 bug) ────


@pytest.mark.property
@given(
    a_mark_frame=st.one_of(st.none(), st.integers(min_value=0, max_value=100000)),
    a_mark_label=_A_LABEL,
    b_mark_frame=st.one_of(st.none(), st.integers(min_value=0, max_value=100000)),
    b_mark_label=_B_LABEL,
)
def test_actor_label_lines_are_isolated(
    a_mark_frame,
    a_mark_label,
    b_mark_frame,
    b_mark_label,
):
    """Exactly one line starts with each actor's label — labels don't bleed."""
    # Constrain labels to be different so we can unambiguously identify lines
    assume(a_mark_label != b_mark_label or a_mark_label in {"A out", "A in"})
    # Labels might coincide ("A out" vs "B in" - always distinct)
    # So no assume needed for the standard pair
    lines = _make_lines(
        a_mark_frame=a_mark_frame,
        a_mark_label=a_mark_label,
        b_mark_frame=b_mark_frame,
        b_mark_label=b_mark_label,
    )
    a_line_candidates = [l for l in lines if l.startswith(a_mark_label + ":")]
    b_line_candidates = [l for l in lines if l.startswith(b_mark_label + ":")]
    assert len(a_line_candidates) == 1, (
        f"expected exactly 1 line starting with {a_mark_label + ':'!r}, got "
        f"{a_line_candidates} from {lines}"
    )
    assert len(b_line_candidates) == 1, (
        f"expected exactly 1 line starting with {b_mark_label + ':'!r}, got "
        f"{b_line_candidates} from {lines}"
    )

    a_line = a_line_candidates[0]
    b_line = b_line_candidates[0]
    expected_a = f"f={a_mark_frame}" if a_mark_frame is not None else "f=?"
    expected_b = f"f={b_mark_frame}" if b_mark_frame is not None else "f=?"
    assert a_line.endswith(expected_a), f"A-actor line {a_line!r} should end with {expected_a!r}"
    assert b_line.endswith(expected_b), f"B-actor line {b_line!r} should end with {expected_b!r}"


@pytest.mark.property
@given(
    a_mark_frame=st.integers(min_value=0, max_value=100000),
    b_mark_frame=st.integers(min_value=0, max_value=100000),
)
def test_no_f_none_in_output(a_mark_frame, b_mark_frame):
    lines = _make_lines(a_mark_frame=a_mark_frame, b_mark_frame=b_mark_frame)
    assert "f=None" not in "\n".join(lines)


# ── Example-based regression: _sanitize_single_line handles all breaks ─
# Property test found this via Hypothesis: str.splitlines() recognizes
# 11 line-break chars, not just \n and \r. A naive replace misses the
# exotic ones (\v, \f, \x1c, \x1d, \x1e, \x85, \u2028, \u2029).
# Lock them with a hardcoded parameterized test so a future weakening of
# the property strategy doesn't silently regress.


@pytest.mark.property
@pytest.mark.parametrize(
    "line_break",
    [
        "\n",
        "\r",
        "\v",
        "\f",
        "\x1c",
        "\x1d",
        "\x1e",
        "\x85",
        "\u2028",
        "\u2029",
    ],
)
def test_sanitize_handles_each_line_break_char(line_break):
    """Every char str.splitlines() splits on must be neutralized."""
    from src.analysis.visualization.pet_samples import _build_metadata_lines

    # Inject the line break into each text field
    for field in ("video_source", "grid_cell", "conflict_type"):
        kwargs = dict(
            video_source="v",
            frame_idx=1,
            grid_cell="g",
            conflict_type="t",
            track_a=1,
            track_b=2,
            a_mark_frame=1,
            a_mark_label="A out",
            b_mark_frame=2,
            b_mark_label="B in",
        )
        kwargs[field] = f"prefix{line_break}suffix"
        lines = _build_metadata_lines(**kwargs)
        assert len(lines) == 7
        for i, line in enumerate(lines):
            assert len(line.splitlines()) == 1, (
                f"field {field!r} with {line_break!r} produced multi-line "
                f"output on line {i}: {line!r}"
            )


@pytest.mark.property
@pytest.mark.parametrize(
    "line_break",
    [
        "\n",
        "\r",
        "\v",
        "\f",
        "\x1c",
        "\x1d",
        "\x1e",
        "\x85",
        "\u2028",
        "\u2029",
    ],
)
def test_verbatim_preserved_when_no_line_break(line_break):
    """Sanitization replaces the separator with a space, preserving text."""
    from src.analysis.visualization.pet_samples import _build_metadata_lines

    lines = _build_metadata_lines(
        video_source=f"a{line_break}b",
        frame_idx=1,
        grid_cell="g",
        conflict_type="t",
        track_a=1,
        track_b=2,
        a_mark_frame=1,
        a_mark_label="A out",
        b_mark_frame=2,
        b_mark_label="B in",
    )
    assert lines[0] == "Video: a b", f"expected 'Video: a b', got {lines[0]!r}"
