from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple

# Optional: integrate with core NNDS types if present
try:
    from nnds.core.types import (
        WorldSample as CoreWorldSample,
        Interval as CoreInterval,
        PETEvent as CorePETEvent,
        PETSummary as CorePETSummary,
    )
    CORE_TYPES_AVAILABLE = True
except ImportError:
    CORE_TYPES_AVAILABLE = False


# ---------------------------------------------------------------------
# Local dataclasses (used if core types not available)
# ---------------------------------------------------------------------
@dataclass
class WorldSample:
    t: float  # seconds
    x: float  # meters
    y: float  # meters

    def __repr__(self) -> str:
        return f"WorldSample(t={self.t:.2f}, x={self.x:.1f}, y={self.y:.1f})"


@dataclass
class Interval:
    obj_id: int
    cell_id: Any
    t_enter: float
    t_exit: float
    world_samples: List[WorldSample]


@dataclass
class PETEvent:
    obj_i: int
    obj_j: int
    cell_id: Any
    t_exit_i: float
    t_enter_j: float
    pet: float
    world_traj_i: List[WorldSample]
    world_traj_j: List[WorldSample]
    severity: str  # "critical" / "moderate" / "safe"

    @property
    def is_critical(self) -> bool:
        return self.severity == "critical"

    @property
    def is_conflict(self) -> bool:
        return self.severity in ("critical", "moderate")

    @property
    def time_gap(self) -> float:
        """Time gap between exit and entry (alias for pet)."""
        return self.pet

    def __repr__(self) -> str:
        return (
            f"PETEvent({self.obj_i}->{self.obj_j}, "
            f"cell={self.cell_id}, pet={self.pet:.2f}s, "
            f"severity={self.severity})"
        )


@dataclass
class PETSummary:
    count: int
    min_pet: Optional[float]
    max_pet: Optional[float]
    mean_pet: Optional[float]
    p5: Optional[float]
    p50: Optional[float]
    p95: Optional[float]
    n_critical: int
    n_moderate: int
    n_safe: int


# Bind to core types if present
if CORE_TYPES_AVAILABLE:
    WorldSampleType = CoreWorldSample
    IntervalType = CoreInterval
    PETEventType = CorePETEvent
    PETSummaryType = CorePETSummary
else:
    WorldSampleType = WorldSample
    IntervalType = Interval
    PETEventType = PETEvent
    PETSummaryType = PETSummary


# ---------------------------------------------------------------------
# Trajectory logger with validation + stats + optional downsampling
# ---------------------------------------------------------------------
class TrajectoryLogger:
    """
    Logs per-frame cell occupancy and builds continuous intervals.

    Validates FPS, supports optional downsampling of world trajectories
    to reduce memory usage, and exposes basic stats.
    """

    def __init__(self, fps: float, downsample_every: int = 1, max_frame_gap: int = 5) -> None:
        """
        Args:
            fps: Frames per second, must be > 0.
            downsample_every: Keep every k-th world sample (1 = keep all).
            max_frame_gap: If a track is not observed in the same cell for
                more than this many consecutive frames (e.g. occlusion),
                the interval is closed rather than treated as continuous
                presence spanning the gap.
        """
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        if downsample_every < 1:
            raise ValueError(f"downsample_every must be >= 1, got {downsample_every}")
        if max_frame_gap < 1:
            raise ValueError(f"max_frame_gap must be >= 1, got {max_frame_gap}")

        self.fps: float = float(fps)
        self.downsample_every: int = int(downsample_every)
        self.max_frame_gap: int = int(max_frame_gap)
        # track_id -> list[(frame_idx, cell_id, world_x, world_y)]
        self.tracks: DefaultDict[
            int, List[Tuple[int, Any, Optional[float], Optional[float]]]
        ] = defaultdict(list)

    def log(
        self,
        track_id: int,
        frame_idx: int,
        cell_id: Any,
        world_x: Optional[float] = None,
        world_y: Optional[float] = None,
    ) -> None:
        """
        Log a single observation for a track.

        Coordinates are not range-checked here, but you can add project-specific
        bounds if needed (e.g., enforce BEV limits).
        """
        tid = int(track_id)
        fi = int(frame_idx)
        self.tracks[tid].append((fi, cell_id, world_x, world_y))

    def build_intervals(self) -> List[IntervalType]:
        """
        Build continuous occupancy intervals per (track_id, cell_id),
        with optional downsampling of world samples.

        Returns:
            List of IntervalType.
        """
        intervals: List[IntervalType] = []

        for obj_id, samples in self.tracks.items():
            samples.sort(key=lambda x: x[0])

            prev_cell: Any = None
            prev_frame: Optional[int] = None
            start_frame: Optional[int] = None
            world_samples: List[WorldSampleType] = []
            sample_counter = 0

            for frame_idx, cell_id, wx, wy in samples:
                fi = int(frame_idx)

                if prev_cell is None:
                    prev_cell = cell_id
                    prev_frame = fi
                    start_frame = fi
                    if wx is not None and wy is not None:
                        if sample_counter % self.downsample_every == 0:
                            world_samples.append(
                                WorldSampleType(t=fi / self.fps, x=float(wx), y=float(wy))
                            )
                        sample_counter += 1
                    continue

                gap = fi - prev_frame if prev_frame is not None else 0
                same_cell_continuous = (cell_id == prev_cell) and (gap <= self.max_frame_gap)

                if same_cell_continuous:
                    if wx is not None and wy is not None:
                        if sample_counter % self.downsample_every == 0:
                            world_samples.append(
                                WorldSampleType(t=fi / self.fps, x=float(wx), y=float(wy))
                            )
                        sample_counter += 1
                else:
                    # Close previous interval at the last frame it was
                    # actually observed (not fi - 1, since a gap may exist).
                    if start_frame is not None:
                        end_frame = prev_frame if prev_frame is not None else fi - 1
                        intervals.append(
                            IntervalType(
                                obj_id=obj_id,
                                cell_id=prev_cell,
                                t_enter=start_frame / self.fps,
                                t_exit=end_frame / self.fps,
                                world_samples=list(world_samples),
                            )
                        )

                    # Start new interval
                    prev_cell = cell_id
                    start_frame = fi
                    world_samples = []
                    sample_counter = 0
                    if wx is not None and wy is not None:
                        world_samples.append(
                            WorldSampleType(t=fi / self.fps, x=float(wx), y=float(wy))
                        )
                        sample_counter += 1

                prev_frame = fi

            # Close last interval for this object
            if prev_cell is not None and start_frame is not None and samples:
                end_frame = samples[-1][0]
                intervals.append(
                    IntervalType(
                        obj_id=obj_id,
                        cell_id=prev_cell,
                        t_enter=start_frame / self.fps,
                        t_exit=end_frame / self.fps,
                        world_samples=list(world_samples),
                    )
                )

        return intervals

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about logged data."""
        total_samples = sum(len(samples) for samples in self.tracks.values())
        return {
            "num_tracks": len(self.tracks),
            "total_samples": total_samples,
            "fps": self.fps,
            "downsample_every": self.downsample_every,
            "avg_samples_per_track": total_samples / max(len(self.tracks), 1),
        }


# ---------------------------------------------------------------------
# PET severity and statistics
# ---------------------------------------------------------------------
def classify_pet(
    pet: float,
    critical_threshold: float = 1.5,
    moderate_threshold: float = 3.0,
) -> str:
    """Classify PET into severity levels."""
    if pet < critical_threshold:
        return "critical"
    if pet < moderate_threshold:
        return "moderate"
    return "safe"


def summarize_pet(
    events: Sequence[PETEventType],
    critical_threshold: float = 1.5,
    moderate_threshold: float = 3.0,
) -> PETSummaryType:
    """
    Aggregate PET statistics and severity counts.
    """
    if not events:
        return PETSummaryType(
            count=0,
            min_pet=None,
            max_pet=None,
            mean_pet=None,
            p5=None,
            p50=None,
            p95=None,
            n_critical=0,
            n_moderate=0,
            n_safe=0,
        )

    pets = sorted(ev.pet for ev in events)
    count = len(pets)
    min_pet = pets[0]
    max_pet = pets[-1]
    mean_pet = sum(pets) / count

    def percentile(p: float) -> float:
        idx = max(0, min(count - 1, int(round(p * (count - 1)))))
        return pets[idx]

    p5 = percentile(0.05)
    p50 = percentile(0.50)
    p95 = percentile(0.95)

    n_critical = sum(1 for ev in events if ev.pet < critical_threshold)
    n_moderate = sum(
        1 for ev in events if critical_threshold <= ev.pet < moderate_threshold
    )
    n_safe = count - n_critical - n_moderate

    return PETSummaryType(
        count=count,
        min_pet=min_pet,
        max_pet=max_pet,
        mean_pet=mean_pet,
        p5=p5,
        p50=p50,
        p95=p95,
        n_critical=n_critical,
        n_moderate=n_moderate,
        n_safe=n_safe,
    )


# ---------------------------------------------------------------------
# Fixed PET computation logic (both directions, no duplicates)
# ---------------------------------------------------------------------
def compute_pet(
    intervals: Sequence[IntervalType],
    pet_threshold: float = 2.0,
    critical_threshold: float = 1.5,
    moderate_threshold: float = 3.0,
) -> List[PETEventType]:
    """
    Compute PET events from intervals with input validation and severity.

    This version:
    - Validates pet_threshold.
    - Sorts intervals by t_enter per cell.
    - Checks both A→B and B→A directions.
    - Only considers pairs with j > i to avoid duplicates.
    """
    if pet_threshold <= 0:
        raise ValueError(f"pet_threshold must be positive, got {pet_threshold}")

    pet_events: List[PETEventType] = []
    by_cell: Dict[Any, List[IntervalType]] = {}

    for iv in intervals:
        by_cell.setdefault(iv.cell_id, []).append(iv)

    for cell_id, cell_intervals in by_cell.items():
        # Sort by enter time for efficient processing
        sorted_intervals = sorted(cell_intervals, key=lambda iv: iv.t_enter)
        n = len(sorted_intervals)

        for i in range(n):
            A = sorted_intervals[i]
            for j in range(i + 1, n):  # only j > i to avoid duplicates
                B = sorted_intervals[j]

                if A.obj_id == B.obj_id:
                    # Same object re-entering the same cell is not a conflict
                    # with itself; skip to avoid fabricated self-PET events.
                    continue

                # Case 1: A exits before B enters (A -> B)
                if A.t_exit <= B.t_enter:
                    pet = B.t_enter - A.t_exit
                    if 0.0 < pet <= pet_threshold:
                        severity = classify_pet(
                            pet,
                            critical_threshold=critical_threshold,
                            moderate_threshold=moderate_threshold,
                        )
                        pet_events.append(
                            PETEventType(
                                obj_i=A.obj_id,
                                obj_j=B.obj_id,
                                cell_id=cell_id,
                                t_exit_i=A.t_exit,
                                t_enter_j=B.t_enter,
                                pet=pet,
                                world_traj_i=list(A.world_samples),
                                world_traj_j=list(B.world_samples),
                                severity=severity,
                            )
                        )

                # Case 2: B exits before A enters (B -> A)
                elif B.t_exit <= A.t_enter:
                    pet = A.t_enter - B.t_exit
                    if 0.0 < pet <= pet_threshold:
                        severity = classify_pet(
                            pet,
                            critical_threshold=critical_threshold,
                            moderate_threshold=moderate_threshold,
                        )
                        pet_events.append(
                            PETEventType(
                                obj_i=B.obj_id,
                                obj_j=A.obj_id,
                                cell_id=cell_id,
                                t_exit_i=B.t_exit,
                                t_enter_j=A.t_enter,
                                pet=pet,
                                world_traj_i=list(B.world_samples),
                                world_traj_j=list(A.world_samples),
                                severity=severity,
                            )
                        )

    return pet_events
