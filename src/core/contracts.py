
"""
Production data contracts for NNDS simplified PET outputs.

These Pydantic models mirror the actual columns in
`outputs/*_screened_simplified.csv` and are used to validate rows at runtime.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

ConflictType = Literal["crossing", "head_on", "rear_end", "side_swipe", "other"]
SiteLabel = Literal["GITI", "MRC"]

class PETEventRecord(BaseModel):
    event_id: int = Field(..., ge=0)
    pet: float = Field(..., ge=0)
    frame: int = Field(..., ge=0)
    track_a: int
    track_b: int
    conflict_type: ConflictType
    grid_cell: str
    track_a_entry_frame: int | None = None
    track_a_exit_frame: int | None = None
    track_a_exit_time_sec: float | None = None
    track_b_entry_frame: int | None = None
    track_b_entry_time_sec: float | None = None
    track_b_exit_frame: int | None = None
    site: SiteLabel
    time_of_day_label: str | None = None
    gate_a_entry: str | None = None
    gate_b_entry: str | None = None
    first_track_id: int | None = None
    second_track_id: int | None = None
    first_exit_frame: int | None = None
    first_exit_time_sec: float | None = None
    second_entry_frame: int | None = None
    second_entry_time_sec: float | None = None
    pet_frames: int | None = None
    pet_s: float | None = None
    occupancy_relation: Literal["a_to_b", "b_to_a"] | None = None

    @model_validator(mode="after")
    def check_pet_positive(self):
        if self.pet < 0:
            raise ValueError("PET must be non-negative")
        return self

    @model_validator(mode="after")
    def check_track_order(self):
        if self.track_a == self.track_b:
            raise ValueError("track_a and track_b must differ")
        return self
