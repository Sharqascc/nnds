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

__all__ = [
    "Detection",
    "PETEvent",
    "TrackingState",
    "TrackingStateMachine",
    "Trajectory",
    "WorldPoint",
    "pure",
]

def pure(func):
    """Mark a function as pure (no side effects, deterministic)."""
    func._pure = True
    return func


class WorldPoint(BaseModel):
    t: float = Field(..., ge=0, description="Time in seconds")
    x: float = Field(..., ge=0, description="Pixel X coordinate (non-negative)")
    y: float = Field(..., ge=0, description="Pixel Y coordinate (non-negative)")


class Trajectory(BaseModel):
    track_id: int = Field(..., ge=0)
    points: list[WorldPoint] = Field(..., min_length=2)
    actor_type: str | None = None
    source: str | None = None

    @model_validator(mode="after")
    def points_must_be_ordered(self):
        times = [p.t for p in self.points]
        if any(times[i] >= times[i + 1] for i in range(len(times) - 1)):
            raise ValueError("Trajectory points must be strictly increasing in time")
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
        if not (self.x1 <= self.cx <= self.x2 and self.y1 <= self.cy <= self.y2):
            raise ValueError("Center (cx, cy) must lie within bounding box")
        return self


class PETEvent(BaseModel):
    event_id: int
    pet: float = Field(..., ge=0)
    frame: int = Field(..., ge=0)
    track_a: int = Field(..., ge=0)
    track_b: int = Field(..., ge=0)
    conflict_type: Literal["crossing", "head_on", "rear_end", "side_swipe", "other"]
    grid_cell: str
    track_a_exit_frame: int
    track_b_entry_frame: int
    site: str

    @model_validator(mode="after")
    def distinct_tracks(self):
        if self.track_a == self.track_b:
            raise ValueError("track_a and track_b must be different")
        return self


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
