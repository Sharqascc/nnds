"""Interval-based Post-Encroachment Time (PET) calculation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

PetStatus = Literal["sequential", "overlap", "invalid"]


@dataclass(frozen=True)
class PetResult:
    """Structured PET result preserving overlap information."""

    pet_s: float | None
    pet_status: PetStatus
    first_actor: str | None
    second_actor: str | None
    overlap_duration_s: float


def _validate_interval(entry: float, exit: float, actor: str) -> None:
    if not isfinite(entry) or not isfinite(exit):
        raise ValueError(f"{actor} interval contains NaN or infinity")

    if entry > exit:
        raise ValueError(f"{actor}_entry must be <= {actor}_exit")


def compute_pet_from_intervals(
    a_entry: float,
    a_exit: float,
    b_entry: float,
    b_exit: float,
) -> PetResult:
    """Compute PET from two conflict-zone occupancy intervals."""
    _validate_interval(a_entry, a_exit, "a")
    _validate_interval(b_entry, b_exit, "b")

    if a_exit <= b_entry:
        return PetResult(
            pet_s=b_entry - a_exit,
            pet_status="sequential",
            first_actor="a",
            second_actor="b",
            overlap_duration_s=0.0,
        )

    if b_exit <= a_entry:
        return PetResult(
            pet_s=a_entry - b_exit,
            pet_status="sequential",
            first_actor="b",
            second_actor="a",
            overlap_duration_s=0.0,
        )

    overlap_duration = min(a_exit, b_exit) - max(a_entry, b_entry)

    return PetResult(
        pet_s=None,
        pet_status="overlap",
        first_actor=None,
        second_actor=None,
        overlap_duration_s=max(0.0, overlap_duration),
    )


def compute_pet_from_frames(
    a_entry_frame: int,
    a_exit_frame: int,
    b_entry_frame: int,
    b_exit_frame: int,
    fps: float,
) -> PetResult:
    """Compute PET from frame-based occupancy windows."""
    if not isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a finite positive number")

    return compute_pet_from_intervals(
        a_entry_frame / fps,
        a_exit_frame / fps,
        b_entry_frame / fps,
        b_exit_frame / fps,
    )
