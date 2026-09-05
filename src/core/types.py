from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import pandas as pd

# ===== Low-level geometric primitives =====


@dataclass(frozen=True)
class WorldPoint:
    t: float  # seconds
    x: float  # meters
    y: float  # meters

    def __post_init__(self):
        for coord in (self.t, self.x, self.y):
            if not math.isfinite(coord):
                raise ValueError(f"WorldPoint coordinates must be finite, got {coord}")


@dataclass(frozen=True)
class Trajectory:
    track_id: int
    points: tuple[WorldPoint, ...]
    actor_type: str | None = None  # e.g., "pedestrian", "car", etc.
    source: str | None = None  # e.g., "sam3", "gt", etc.

    def __post_init__(self):
        if not self.points:
            return  # allow empty trajectory; duration returns 0.0
        times = [p.t for p in self.points]
        if any(times[i] >= times[i + 1] for i in range(len(times) - 1)):
            raise ValueError("Trajectory points must be strictly increasing in time")

    @property
    def duration(self) -> float:
        if not self.points:  # pragma: no cover
            return 0.0  # pragma: no cover
        return self.points[-1].t - self.points[0].t  # pragma: no cover


# ===== PET / conflict events =====


@dataclass(frozen=True)
class PETEvent:
    event_id: int
    pet: float  # seconds
    track_a: int
    track_b: int
    conflict_type: str  # e.g. "CELL_C_1"
    world_traj_i: Trajectory
    world_traj_j: Trajectory
    frame: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not math.isfinite(self.pet):
            raise ValueError("PET value must be finite")
        if self.track_a == self.track_b:
            raise ValueError("track_a and track_b must be different")


# ===== Diffusion training / sampling =====


@dataclass(frozen=True)
class TrajectoryBatch:
    """
    Canonical representation for diffusion model training / sampling.
    Shapes are *logical* here; code can store them as np.ndarray or torch.Tensor.
    """

    inputs: Any  # shape: (B, T_in, D)
    targets: Any  # shape: (B, T_out, D)
    meta: Mapping[str, Any]
    fps: float

    def __post_init__(self):
        # Validate that inputs and targets have shape attribute and at least 2 dims
        for name, obj in [("inputs", self.inputs), ("targets", self.targets)]:
            if not hasattr(obj, "shape") or len(obj.shape) < 2:
                raise ValueError(f"{name} must have shape (B, T, D) with at least 2 dimensions")
            if not all(isinstance(dim, int) and dim >= 0 for dim in obj.shape):
                raise ValueError(f"{name} shape dimensions must be non-negative integers")
        # Validate batch size compatibility
        if self.inputs.shape[0] != self.targets.shape[0]:
            raise ValueError("inputs and targets must have the same batch size")
        if not math.isfinite(self.fps):
            raise ValueError("fps must be finite")

    @property
    def batch_size(self) -> int:
        return int(self.inputs.shape[0])  # pragma: no cover

    @property
    def input_length(self) -> int:
        return int(self.inputs.shape[1])  # pragma: no cover

    @property
    def target_length(self) -> int:
        return int(self.targets.shape[1])  # pragma: no cover


# ===== Protocols (interfaces) =====


class PETDataFrameLike(Protocol):
    """
    Minimal interface that PET analysis / viz code expects from a PET dataset.
    This lets you use pandas.DataFrame, polars.DataFrame, etc.
    """

    def __getitem__(self, key: str) -> Sequence[Any]: ...

    @property
    def columns(self) -> Any: ...

    def to_pandas(self) -> pd.DataFrame: ...


class DiffusionDatasetLike(Protocol):
    """
    Interface for a dataset that can feed the diffusion model.
    """

    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> TrajectoryBatch: ...
