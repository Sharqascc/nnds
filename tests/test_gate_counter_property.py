
import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.analysis.gate_counter import RobustTracker, VirtualGate


# ---------- VirtualGate.check_crossing invariants ----------
def gate_strategy():
    return st.builds(
        VirtualGate,
        name=st.text(min_size=1, max_size=5),
        p1=st.tuples(st.integers(0, 50), st.integers(0, 50)),
        p2=st.tuples(st.integers(0, 50), st.integers(0, 50)),
        entry_side=st.sampled_from(["left", "right"]),
    ).filter(lambda g: g.p1 != g.p2)

@given(gate_strategy(), st.tuples(st.floats(0, 100), st.floats(0, 100)), st.tuples(st.floats(0, 100), st.floats(0, 100)), st.integers(1, 100), st.integers(0, 500))
def test_check_crossing_returns_valid_status(gate, prev, curr, track_id, frame_idx):
    status = gate.check_crossing(prev, curr, track_id, frame_idx)
    assert status in {None, "entry", "exit"}

@given(gate_strategy(), st.tuples(st.floats(0, 100), st.floats(0, 100)), st.tuples(st.floats(0, 100), st.floats(0, 100)), st.integers(1, 100), st.integers(0, 500))
def test_check_crossing_counts_consistent(gate, prev, curr, track_id, frame_idx):
    before_entry = gate.entry_count
    before_exit = gate.exit_count
    status = gate.check_crossing(prev, curr, track_id, frame_idx)
    if status == "entry":
        assert gate.entry_count == before_entry + 1
    elif status == "exit":
        assert gate.exit_count == before_exit + 1
    else:
        assert gate.entry_count == before_entry and gate.exit_count == before_exit

# ---------- RobustTracker invariants ----------
@given(st.lists(st.dictionaries(keys=st.text(), values=st.floats(), min_size=1, max_size=5), max_size=10), st.integers(0, 1000))
def test_robust_tracker_no_negative_missed(dets, frame_idx):
    tracker = RobustTracker(max_missing=30)
    out = tracker.update(dets, frame_idx=frame_idx)
    for tid, track in out.items():
        assert track.get("missed", 0) >= 0
        assert track.get("last_seen_frame", frame_idx) >= 0
