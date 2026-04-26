from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, Tuple, Mapping, Any, Optional
import numpy as np
import pandas as pd


# ===== Low-level geometric primitives =====

@dataclass(frozen=True)
class WorldPoint:
    t: float          # seconds
    x: float          # meters
    y: float          # meters


@dataclass(frozen=True)
class Trajectory:
    track_id: int
    points: Tuple[WorldPoint, ...]
    actor_type: Optional[str] = None     # e.g., "pedestrian", "car", etc.
    source: Optional[str] = None         # e.g., "sam3", "gt", etc.

    @property
    def duration(self) -> float:
        if not self.points:
            return 0.0
        return self.points[-1].t - self.points[0].t


# ===== PET / conflict events =====

@dataclass(frozen=True)
class PETEvent:
    event_id: int
    pet: float                       # seconds
    track_a: int
    track_b: int
    conflict_type: str               # e.g. "CELL_C_1"
    world_traj_i: Trajectory
    world_traj_j: Trajectory
    frame: Optional[int] = None
    metadata: Mapping[str, Any] = None


# ===== Diffusion training / sampling =====

@dataclass(frozen=True)
class TrajectoryBatch:
    """
    Canonical representation for diffusion model training / sampling.
    Shapes are *logical* here; code can store them as np.ndarray or torch.Tensor.
    """
    inputs: Any        # shape: (B, T_in, D)
    targets: Any       # shape: (B, T_out, D)
    meta: Mapping[str, Any]
    fps: float

    @property
    def batch_size(self) -> int:
        return int(self.inputs.shape[0])

    @property
    def input_length(self) -> int:
        return int(self.inputs.shape[1])

    @property
    def target_length(self) -> int:
        return int(self.targets.shape[1])


# ===== Protocols (interfaces) =====

class PETDataFrameLike(Protocol):
    """
    Minimal interface that PET analysis / viz code expects from a PET dataset.
    This lets you use pandas.DataFrame, polars.DataFrame, etc.
    """

    def __getitem__(self, key: str) -> Sequence[Any]:
        ...

    @property
    def columns(self) -> Any:
        ...

    def to_pandas(self) -> pd.DataFrame:
        ...


class DiffusionDatasetLike(Protocol):
    """
    Interface for a dataset that can feed the diffusion model.
    """

    def __len__(self) -> int:
        ...

    def __getitem__(self, idx: int) -> TrajectoryBatch:
        ...
