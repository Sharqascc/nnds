
"""
Production data contracts for NNDS simplified PET outputs.

These Pydantic models mirror the actual columns in
`outputs/*_screened_simplified.csv` and are used to validate rows at runtime.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

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


# ----------------------------------------------------------------------
# PET Summary contracts
# ----------------------------------------------------------------------
class RiskLevelSummary(BaseModel):
    count: int = Field(..., ge=0)
    percentage: float = Field(..., ge=0, le=100)

class ConflictRateSummary(BaseModel):
    count: int = Field(..., ge=0)
    percentage: float = Field(..., ge=0, le=100)
    per_1000_events: float = Field(..., ge=0)

class PETSummaryRiskSummary(BaseModel):
    critical: RiskLevelSummary
    serious: RiskLevelSummary
    moderate: RiskLevelSummary
    safe: RiskLevelSummary
    conflict_rate: ConflictRateSummary

class PETSummaryBasicStats(BaseModel):
    count: int = Field(..., ge=0)
    mean: float
    std: float | None = None
    sem: float | None = None
    min: float
    q25: float
    median: float
    q75: float
    max: float
    iqr: float
    cv: float | None = None
    skew: float | None = None
    kurtosis: float | None = None
    ci_mean_lower: float | None = None
    ci_mean_upper: float | None = None
    ci_level: float | None = None
    p1: float
    p5: float
    p10: float
    p90: float
    p95: float
    p99: float


class PETUncertaintyContract(BaseModel):
    nominal_pet: float = Field(..., ge=0)
    uncertainty_std: float = Field(..., ge=0)
    error_sources: dict[str, float]

    @model_validator(mode="after")
    def check_error_sources(self):
        if not self.error_sources:
            raise ValueError("error_sources must not be empty")
        if any(v < 0 for v in self.error_sources.values()):
            raise ValueError("error_sources values must be non-negative")
        return self


class UQAnalysisContract(BaseModel):
    metric_name: str
    passed: bool
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    method: str

    @model_validator(mode="after")
    def check_passed_consistency(self):
        if not self.passed and not self.errors:
            raise ValueError("If passed is False, errors must not be empty")
        return self



class StatisticalTestResultContract(BaseModel):
    test: str
    alpha: float | None = None
    passed: bool | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    type: str | None = None
    statistics: dict[str, Any] = Field(default_factory=dict)
    test_statistics: dict[str, Any] = Field(default_factory=dict)
    assumptions: dict[str, Any] = Field(default_factory=dict)
    summary: str | None = None

    @model_validator(mode="after")
    def check_passed_consistency(self):
        if self.passed is False and not self.errors:
            raise ValueError("If passed is False, errors must not be empty")
        return self

