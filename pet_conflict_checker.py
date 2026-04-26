#!/usr/bin/env python
"""
PET Conflict Checker - Core safety analysis module.

Integrates with the NNDS pipeline and traffic_analyzer.py by providing:

- Grid-based PET computation hooks.
- Trajectory pairing utilities.
- ROI-based filtering.
- Conflict detection on PET CSVs produced by the pipeline.

It intentionally reuses PET values already computed in the pipeline CSV
instead of re-implementing low-level PET math here.

Now also exposes typed core structures:
- PETEvent
- Trajectory
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Dict, Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from nnds.core.types import PETEvent, Trajectory, WorldPoint  # <— NEW import

PET_COLUMN_CANDIDATES = ["pet", "pet_sec", "true_pet_sec", "pet_sample_sec"]


def _find_pet_column(df: pd.DataFrame) -> Optional[str]:
    """Find the most likely PET column in a DataFrame."""
    for col in PET_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    return None


def compute_pet(
    times_a: Iterable[float],
    times_b: Iterable[float],
) -> float:
    """
    Compute PET given the times at which two agents pass a conflict point.

    This is the *conceptual* PET, not a full trajectory-based solver.
    """
    ta = np.array(list(times_a), dtype=float)
    tb = np.array(list(times_b), dtype=float)

    if ta.size == 0 or tb.size == 0:
        return np.inf

    diff_matrix = np.abs(ta[:, None] - tb[None, :])
    return float(diff_matrix.min())


def compute_grid_pet(
    grid_a: np.ndarray,
    grid_b: np.ndarray,
    fps: float,
) -> float:
    """
    Compute PET from two occupancy grids over time.
    """
    if grid_a.shape != grid_b.shape:
        raise ValueError("grid_a and grid_b must have the same shape (T, H, W).")

    T = grid_a.shape[0]
    occ_a = grid_a.reshape(T, -1).any(axis=1)
    occ_b = grid_b.reshape(T, -1).any(axis=1)

    t_a = np.where(occ_a)[0]
    t_b = np.where(occ_b)[0]

    if t_a.size == 0 or t_b.size == 0:
        return np.inf

    diff_matrix = np.abs(t_a[:, None] - t_b[None, :])
    frame_gap = diff_matrix.min()
    return float(frame_gap / fps)


def filter_by_roi(
    df: pd.DataFrame,
    roi: Dict[str, float],
    x_col: str = "x",
    y_col: str = "y",
) -> pd.DataFrame:
    """
    Filter events/trajectories by region-of-interest (ROI).
    """
    required_keys = {"xmin", "xmax", "ymin", "ymax"}
    if not required_keys.issubset(roi.keys()):
        raise ValueError(f"ROI must contain keys: {sorted(required_keys)}")

    mask = (
        (df[x_col] >= roi["xmin"])
        & (df[x_col] <= roi["xmax"])
        & (df[y_col] >= roi["ymin"])
        & (df[y_col] <= roi["ymax"])
    )
    return df.loc[mask].copy()


def get_trajectory_pairs(
    df: pd.DataFrame,
    id_col: str = "track_id",
    frame_col: str = "frame",
) -> List[Tuple[int, int]]:
    """
    Construct candidate trajectory pairs for conflict analysis.
    """
    pairs: set[Tuple[int, int]] = set()
    grouped = df.groupby(frame_col)[id_col]

    for _, ids in grouped:
        ids_list = sorted(set(ids.tolist()))
        for i in range(len(ids_list)):
            for j in range(i + 1, len(ids_list)):
                pairs.add((ids_list[i], ids_list[j]))

    return sorted(pairs)


@dataclass
class ConflictResult:
    id_a: Any
    id_b: Any
    pet: float
    frame_start: Optional[int] = None
    frame_end: Optional[int] = None
    extra: Dict[str, Any] = None


def detect_conflicts(
    df: pd.DataFrame,
    pet_threshold: float = 3.0,
    pet_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Detect traffic conflicts from PET event data.

    Returns:
        DataFrame of conflict events with an 'is_conflict' flag.
    """
    if pet_col is None:
        pet_col = _find_pet_column(df)
    if pet_col is None:
        print("⚠️ No PET column found; cannot detect conflicts.")
        return pd.DataFrame(columns=list(df.columns) + ["is_conflict"])

    conflicts = df[df[pet_col] <= pet_threshold].copy()
    conflicts["is_conflict"] = True
    return conflicts


# =====================================================================
# NEW: Bridge between CSV rows and PETEvent / Trajectory dataclasses
# =====================================================================

def _row_to_trajectory(
    traj_data: Any,
    track_id: int,
    actor_type: Optional[str] = None,
    source: Optional[str] = None,
) -> Trajectory:
    """
    Convert a serialized trajectory (e.g. list of [t,x,y] or similar)
    into a Trajectory dataclass with WorldPoint entries.
    """
    # Expect something list-like of (t, x, y); keep it forgiving.
    points: List[WorldPoint] = []
    if traj_data is None:
        return Trajectory(track_id=track_id, points=tuple(points), actor_type=actor_type, source=source)

    for p in traj_data:
        if len(p) < 3:
            continue
        t, x, y = float(p[0]), float(p[1]), float(p[2])
        points.append(WorldPoint(t=t, x=x, y=y))

    return Trajectory(
        track_id=track_id,
        points=tuple(points),
        actor_type=actor_type,
        source=source,
    )


def dataframe_to_pet_events(
    df: pd.DataFrame,
    pet_col: Optional[str] = None,
    traj_i_col: str = "world_traj_i",
    traj_j_col: str = "world_traj_j",
    event_id_col: str = "event_id",
    track_a_col: str = "track_a",
    track_b_col: str = "track_b",
    conflict_type_col: str = "conflict_type",
    frame_col: str = "frame",
) -> List[PETEvent]:
    """
    Convert a PET events DataFrame (pipeline CSV) into typed PETEvent objects.

    Any missing optional columns are handled gracefully.
    """
    if pet_col is None:
        pet_col = _find_pet_column(df)
    if pet_col is None:
        raise ValueError("No PET column found; cannot convert to PETEvent objects.")

    events: List[PETEvent] = []

    for _, row in df.iterrows():
        pet_value = float(row[pet_col])

        event_id = int(row[event_id_col]) if event_id_col in df.columns else -1
        track_a = int(row[track_a_col]) if track_a_col in df.columns else -1
        track_b = int(row[track_b_col]) if track_b_col in df.columns else -1
        conflict_type = (
            str(row[conflict_type_col]) if conflict_type_col in df.columns else "UNKNOWN"
        )
        frame = int(row[frame_col]) if frame_col in df.columns and not pd.isna(row[frame_col]) else None

        traj_i_data = row.get(traj_i_col, None)
        traj_j_data = row.get(traj_j_col, None)

        traj_i = _row_to_trajectory(traj_i_data, track_id=track_a, source="pipeline_csv")
        traj_j = _row_to_trajectory(traj_j_data, track_id=track_b, source="pipeline_csv")

        # Metadata: keep any extra cols
        metadata: Dict[str, Any] = {}
        for col in df.columns:
            if col in {
                pet_col,
                event_id_col,
                track_a_col,
                track_b_col,
                conflict_type_col,
                frame_col,
                traj_i_col,
                traj_j_col,
            }:
                continue
            metadata[col] = row[col]

        events.append(
            PETEvent(
                event_id=event_id,
                pet=pet_value,
                track_a=track_a,
                track_b=track_b,
                conflict_type=conflict_type,
                world_traj_i=traj_i,
                world_traj_j=traj_j,
                frame=frame,
                metadata=metadata,
            )
        )

    return events


class PETConflictChecker:
    """
    Main conflict detection class.

    Designed to be safely importable and usable from traffic_analyzer.py
    and other pipeline components.
    """

    def __init__(self, pet_threshold: float = 3.0):
        self.pet_threshold = pet_threshold

    # --- High-level CSV API -------------------------------------------------

    def detect_from_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Load a PET CSV and return only conflict events as a DataFrame.
        """
        df = pd.read_csv(csv_path)
        return detect_conflicts(df, pet_threshold=self.pet_threshold)

    def detect_from_csv_as_events(self, csv_path: str) -> List[PETEvent]:
        """
        Load a PET CSV and return conflicts as a list of PETEvent dataclasses.
        """
        df = pd.read_csv(csv_path)
        conflicts_df = detect_conflicts(df, pet_threshold=self.pet_threshold)
        return dataframe_to_pet_events(conflicts_df)

    # --- Pipeline integration hooks (safe stubs) ----------------------------

    def extract_trajectories(
        self,
        df: pd.DataFrame,
        id_col: str = "track_id",
        frame_col: str = "frame",
        x_col: str = "x",
        y_col: str = "y",
    ) -> Dict[Any, pd.DataFrame]:
        """
        Extract trajectories from a frame-wise DataFrame.
        """
        trajs: Dict[Any, pd.DataFrame] = {}
        for tid, sub in df.groupby(id_col):
            trajs[tid] = sub.sort_values(frame_col)[[frame_col, x_col, y_col]]
        return trajs

    def get_trajectory_pairs(
        self,
        df: pd.DataFrame,
        id_col: str = "track_id",
        frame_col: str = "frame",
    ) -> List[Tuple[int, int]]:
        """Instance wrapper around module-level get_trajectory_pairs()."""
        return get_trajectory_pairs(df, id_col=id_col, frame_col=frame_col)

    def filter_by_roi(
        self,
        df: pd.DataFrame,
        roi: Dict[str, float],
        x_col: str = "x",
        y_col: str = "y",
    ) -> pd.DataFrame:
        """Instance wrapper around module-level filter_by_roi()."""
        return filter_by_roi(df, roi=roi, x_col=x_col, y_col=y_col)

    # --- Optional high-level video hook ------------------------------------

    def process_video(
        self,
        video_path: str,
        sam3_weights: str,
    ) -> pd.DataFrame:
        """
        Placeholder hook for video -> conflicts.

        Recommended pattern:
            1) Run traffic_analyzer.py externally to produce a PET CSV.
            2) Call detect_from_csv() on that CSV.
        """
        print(
            "⚠️ PETConflictChecker.process_video is a stub.\n"
            "   Use traffic_analyzer.py to generate PET CSV, then\n"
            "   PETConflictChecker.detect_from_csv(csv_path) for conflicts."
        )
        return pd.DataFrame()
