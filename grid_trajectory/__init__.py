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

# Grid structures
from .spatial_grid import SpatialGrid, OUT_OF_BOUNDS_CELL  # type: ignore[import]

# PET core utilities
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

# SAM3 + grid integration (updated)
from .sam3_grid_pet import (
    build_grid_trajectories,
    extract_pet_from_sam3,
    SAM3GridPETResult,
    run_sam3_grid_pet,
    run_sam3_grid_pet_batch,
)

# High-level safety analysis
from .trajectory_safety_analyzer import TrajectorySafetyAnalyzer

__all__: List[str] = [
    # Types
    "WorldSampleType",
    "IntervalType",
    "PETEventType",
    "PETSummaryType",
    "SAM3GridPETResult",
    # Grid / integration
    "SpatialGrid",
    "OUT_OF_BOUNDS_CELL",
    "TrajectoryLogger",
    "build_grid_trajectories",
    "extract_pet_from_sam3",
    "run_sam3_grid_pet",
    "run_sam3_grid_pet_batch",
    "TrajectorySafetyAnalyzer",
    # PET utilities
    "compute_pet",
    "summarize_pet",
    "classify_pet",
]
