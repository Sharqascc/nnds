
import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.pipeline.tracking_state import TrackState, TrackStateMachine

# Generate any state
state_st = st.sampled_from(list(TrackState))

@given(state_st, state_st)
def test_can_transition_is_bool(initial, target):
    machine = TrackStateMachine(initial)
    result = machine.can_transition(target)
    assert isinstance(result, bool)

@given(state_st, state_st)
def test_transition_allowed_updates_state(initial, target):
    machine = TrackStateMachine(initial)
    if machine.can_transition(target):
        machine.transition(target)
        assert machine.state == target
    else:
        with pytest.raises(ValueError):
            machine.transition(target)
        # State should remain unchanged after failed transition
        assert machine.state == initial

@given(state_st, state_st)
def test_transition_disallowed_raises_and_preserves_state(initial, target):
    machine = TrackStateMachine(initial)
    if not machine.can_transition(target):
        before = machine.state
        with pytest.raises(ValueError):
            machine.transition(target)
        assert machine.state == before

def test_default_initial_state_is_detected():
    machine = TrackStateMachine()
    assert machine.state == TrackState.DETECTED

@given(state_st)
def test_exited_has_no_outgoing_transitions(state):
    machine = TrackStateMachine(TrackState.EXITED)
    assert machine.can_transition(state) is False
