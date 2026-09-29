from enum import Enum
from typing import ClassVar


class TrackState(str, Enum):  # noqa: UP042
    DETECTED = "DETECTED"
    TRACKING = "TRACKING"
    MISSED = "MISSED"
    CONFLICT = "CONFLICT"
    EXITED = "EXITED"


class TrackStateMachine:
    _transitions = {
        TrackState.DETECTED: {TrackState.TRACKING, TrackState.EXITED},
        TrackState.TRACKING: {
            TrackState.TRACKING,
            TrackState.MISSED,
            TrackState.CONFLICT,
            TrackState.EXITED,
        },
        TrackState.MISSED: {TrackState.TRACKING, TrackState.EXITED},
        TrackState.CONFLICT: {TrackState.TRACKING, TrackState.EXITED},
        TrackState.EXITED: set(),
    }

    def __init__(self, initial_state: TrackState = TrackState.DETECTED) -> None:
        self.state = initial_state

    def transition(self, new_state: TrackState) -> None:
        if new_state not in self._transitions[self.state]:
            raise ValueError(f"Illegal transition from {self.state} to {new_state}")
        self.state = new_state

    def can_transition(self, new_state):
        return new_state in self._transitions[self.state]
