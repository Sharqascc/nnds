"""
Reasoning-oriented data contracts and state machine for NNDS.

Provides:
- Pydantic models for strict runtime validation
- Explicit tracking state machine
- Pure helper functions
"""

from __future__ import annotations

from enum import Enum, StrEnum
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


def pure(func):
    """Mark a function as pure (no side effects, deterministic)."""
    func._pure = True
    return func


class WorldPoint(BaseModel):
    t: float = Field(..., ge=0, description="Time in seconds")
    x: float = Field(..., description="Pixel X coordinate")
    y: float = Field(..., description="Pixel Y coordinate")


class Trajectory(BaseModel):
    track_id: int = Field(..., ge=0)
    points: list[WorldPoint] = Field(..., min_length=2)
    actor_type: str | None = None
    source: str | None = None

    @model_validator(mode="after")
    def points_must_be_ordered(self):
        times = [p.t for p in self.points]
        if any(times[i] > times[i + 1] for i in range(len(times) - 1)):
            raise ValueError("Trajectory points must be ordered by time")
        return self


class Detection(BaseModel):
    frame: int = Field(..., ge=0)
    x1: float
    y1: float
    x2: float
    y2: float
    cx: float
    cy: float
    cls_id: int
    cls_name: str
    conf: float = Field(..., ge=0, le=1)
    source: str

    @model_validator(mode="after")
    def check_box_valid(self):
        if self.x1 >= self.x2 or self.y1 >= self.y2:
            raise ValueError("Invalid bounding box: x1 < x2 and y1 < y2 required")
        return self


class PETEvent(BaseModel):
    event_id: int
    pet: float = Field(..., ge=0)
    frame: int = Field(..., ge=0)
    track_a: int
    track_b: int
    conflict_type: Literal["crossing", "head_on", "rear_end", "side_swipe", "other"]
    grid_cell: str
    track_a_exit_frame: int
    track_b_entry_frame: int
    site: str


# Tracking state machine
class TrackingState(StrEnum):
    DETECTED = "DETECTED"
    TRACKING = "TRACKING"
    MISSED = "MISSED"
    CONFLICT = "CONFLICT"
    EXITED = "EXITED"


class TrackingStateMachine:
    """Explicit state machine for a tracked object's lifecycle."""

    _transitions = {
        TrackingState.DETECTED: {TrackingState.TRACKING, TrackingState.EXITED},
        TrackingState.TRACKING: {
            TrackingState.TRACKING,
            TrackingState.MISSED,
            TrackingState.CONFLICT,
            TrackingState.EXITED,
        },
        TrackingState.MISSED: {TrackingState.TRACKING, TrackingState.EXITED},
        TrackingState.CONFLICT: {TrackingState.TRACKING, TrackingState.EXITED},
        TrackingState.EXITED: set(),
    }

    def __init__(self, initial_state: TrackingState = TrackingState.DETECTED):
        self.state = initial_state

    def transition(self, new_state: TrackingState) -> None:
        if new_state not in self._transitions[self.state]:
            raise ValueError(f"Illegal transition from {self.state.value} to {new_state.value}")
        self.state = new_state

    def can_transition(self, new_state: TrackingState) -> bool:
        return new_state in self._transitions[self.state]
