from __future__ import annotations

"""
Grid trajectory and PET computation utilities for NNDS.

This package provides:
- Spatial grid definitions
- SAM3 + grid integration
- PET interval logging and conflict extraction
- High-level safety analysis helpers
"""

from typing import List

# Re-export core classes/functions so consumers can do:
#   from grid_trajectory import TrajectoryLogger, compute_pet, summarize_pet, SpatialGrid, ...
from .spatial_grid import SpatialGrid  # type: ignore[import]
from .pet_grid import (
    TrajectoryLogger,
    compute_pet,
    summarize_pet,
    classify_pet,
    WorldSampleType,
    IntervalType,
    PETEventType,
    PETSummaryType,
)
from .sam3_grid_pet import (
    build_grid_trajectories,
    extract_pet_from_sam3,
)
from .trajectory_safety_analyzer import (
    TrajectorySafetyAnalyzer,
)

__all__: List[str] = [
    # Types
    "WorldSampleType",
    "IntervalType",
    "PETEventType",
    "PETSummaryType",
    # Grid / integration
    "SpatialGrid",
    "TrajectoryLogger",
    "build_grid_trajectories",
    "extract_pet_from_sam3",
    "TrajectorySafetyAnalyzer",
    # PET utilities
    "compute_pet",
    "summarize_pet",
    "classify_pet",
]
