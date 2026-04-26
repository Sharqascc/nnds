#!/usr/bin/env python
"""
PET Conflict Checker - Core safety analysis module.

This module is designed to integrate with the existing NNDS pipeline and
traffic_analyzer.py, by providing:

- Grid-based PET computation hooks.
- Trajectory pairing utilities.
- ROI-based filtering.
- Conflict detection on PET CSVs produced by the pipeline.

It intentionally reuses PET values already computed in the pipeline CSV
instead of re-implementing low-level PET math here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Dict, Any, Iterable, Optional

import numpy as np
import pandas as pd


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

    Args:
        times_a: One or more timestamps (seconds) for agent A at conflict point P.
        times_b: One or more timestamps (seconds) for agent B at conflict point P.

    Returns:
        PET in seconds, defined as the absolute time difference between when
        the two agents leave/enter the conflict point.

        If no overlap / times given, returns np.inf.

    Notes:
        In the full NNDS pipeline, PET is typically computed earlier (e.g. in
        grid_trajectory / bev_mapper). This utility exists mainly so that
        traffic_analyzer.py and other modules have a well-defined PET API.
    """
    ta = np.array(list(times_a), dtype=float)
    tb = np.array(list(times_b), dtype=float)

    if ta.size == 0 or tb.size == 0:
        return np.inf

    # PET is the minimum time difference between their visits to the conflict point.
    # Example: A passes at 10.0s, B at 12.5s -> PET = 2.5s
    diff_matrix = np.abs(ta[:, None] - tb[None, :])
    return float(diff_matrix.min())


def compute_grid_pet(
    grid_a: np.ndarray,
    grid_b: np.ndarray,
    fps: float,
) -> float:
    """
    Compute PET from two occupancy grids over time.

    Args:
        grid_a: Boolean or 0/1 array of shape (T, H, W) for agent A.
        grid_b: Boolean or 0/1 array of shape (T, H, W) for agent B.
        fps: Frames per second (to convert frame indices to seconds).

    Returns:
        PET in seconds, or np.inf if there is no sequential encroachment.

    Notes:
        This is a *basic* implementation using grid overlap over time. The
        main, authoritative PET values in the pipeline still come from the
        dedicated safety modules (e.g. grid_trajectory, pet_safety_metrics).
    """
    if grid_a.shape != grid_b.shape:
        raise ValueError("grid_a and grid_b must have the same shape (T, H, W).")

    T = grid_a.shape[0]
    # For each frame, check if either agent occupies any cell (conflict zone union).
    occ_a = grid_a.reshape(T, -1).any(axis=1)
    occ_b = grid_b.reshape(T, -1).any(axis=1)

    # Frames where agent A is in the conflict zone
    t_a = np.where(occ_a)[0]
    t_b = np.where(occ_b)[0]

    if t_a.size == 0 or t_b.size == 0:
        return np.inf

    # PET is time difference between their visits
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

    Args:
        df: DataFrame with at least x_col and y_col.
        roi: Dict with keys 'xmin', 'xmax', 'ymin', 'ymax'.
        x_col: Name of x coordinate column.
        y_col: Name of y coordinate column.

    Returns:
        Filtered DataFrame containing only rows inside the ROI.
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

    Args:
        df: DataFrame with per-frame positions and IDs.
        id_col: Column name for track IDs.
        frame_col: Column name for frame index / time step.

    Returns:
        List of (id_a, id_b) pairs that co-exist in at least one frame.

    Notes:
        This is a heuristic pairing function. Many pipelines use a more
        sophisticated pairing logic in grid_trajectory or pet_safety_metrics.
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

    Args:
        df: DataFrame with PET columns created by the NNDS pipeline
            (e.g. outputs/petevents_bev_*.csv).
        pet_threshold: PET threshold in seconds for conflict classification.
        pet_col: Explicit PET column name; if None, inferred.

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


class PETConflictChecker:
    """
    Main conflict detection class.

    This class is designed to be safely importable and usable from
    traffic_analyzer.py and other pipeline components.

    It relies on PET values pre-computed by the pipeline and does not
    try to duplicate low-level SSM implementations.
    """

    def __init__(self, pet_threshold: float = 3.0):
        self.pet_threshold = pet_threshold

    # --- High-level CSV API -------------------------------------------------

    def detect_from_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Load a PET CSV and return only conflict events.

        Args:
            csv_path: Path to a PET events CSV, typically in outputs/.

        Returns:
            DataFrame with conflicts.
        """
        df = pd.read_csv(csv_path)
        return detect_conflicts(df, pet_threshold=self.pet_threshold)

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

        Args:
            df: DataFrame with per-frame states.
            id_col: Track ID column.
            frame_col: Frame index column.
            x_col, y_col: Position columns.

        Returns:
            Dict mapping track_id -> trajectory DataFrame sorted by frame.
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
        """
        Instance wrapper around module-level get_trajectory_pairs().
        """
        return get_trajectory_pairs(df, id_col=id_col, frame_col=frame_col)

    def filter_by_roi(
        self,
        df: pd.DataFrame,
        roi: Dict[str, float],
        x_col: str = "x",
        y_col: str = "y",
    ) -> pd.DataFrame:
        """
        Instance wrapper around module-level filter_by_roi().
        """
        return filter_by_roi(df, roi=roi, x_col=x_col, y_col=y_col)

    # --- Optional high-level video hook ------------------------------------

    def process_video(
        self,
        video_path: str,
        sam3_weights: str,
    ) -> pd.DataFrame:
        """
        Placeholder hook for video -> conflicts.

        In the current NNDS system, `traffic_analyzer.py` is the main
        entry point from video to PET CSV. To keep this module generic
        and avoid circular imports, we do not call traffic_analyzer.py
        directly here.

        Recommended pattern:
            1) Run traffic_analyzer.py externally to produce a PET CSV.
            2) Call detect_from_csv() on that CSV.

        This method is kept as a stub so that any existing calls won't
        crash; it can be implemented later if you want a single-call
        "video -> conflicts" interface here.

        Args:
            video_path: Path to input video.
            sam3_weights: Path to SAM3 weights.

        Returns:
            Empty DataFrame for now.
        """
        print(
            "⚠️ PETConflictChecker.process_video is a stub.\n"
            "   Use traffic_analyzer.py to generate PET CSV, then\n"
            "   PETConflictChecker.detect_from_csv(csv_path) for conflicts."
        )
        return pd.DataFrame()
