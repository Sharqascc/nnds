from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import logging
import time

import cv2
import pandas as pd

from experimental.contact_point_pipeline import ContactPointPipeline
from grid_trajectory.spatial_grid import SpatialGrid
from grid_trajectory.pet_grid import (
    TrajectoryLogger,
    compute_pet,
    summarize_pet,
    PETEventType,
    IntervalType,
    PETSummaryType,
)
from bev_mapper import BEVMapper


@dataclass
class YOLO26SegGridPETResult:
    fps: float
    frame_count: int
    det_count_total: int
    intervals: List[IntervalType]
    pet_events: List[PETEventType]
    pet_summary: PETSummaryType
    traj_stats: Dict[str, Any]
    processing_time_seconds: float = 0.0


def run_yolo26seg_grid_pet(
    project_root: str | Path,
    video_rel_path: str,
    grid_rel_path: str,
    bev_rel_path: str,
    output_name: str = "yolo26seg_grid_pet_run",
    conf: float = 0.25,
    pet_threshold: float = 2.0,
    max_frames: Optional[int] = None,
    show_progress: bool = True,
    logger: Optional[logging.Logger] = None,
) -> YOLO26SegGridPETResult:
    start_time = time.time()

    root = Path(project_root)
    video_path = root / video_rel_path
    grid_config = root / grid_rel_path
    bev_config = root / bev_rel_path

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not grid_config.is_file():
        raise FileNotFoundError(f"Grid config not found: {grid_config}")
    if not bev_config.is_file():
        raise FileNotFoundError(f"BEV config not found: {bev_config}")

    grid = SpatialGrid(str(grid_config))

    with bev_config.open("r") as f:
        bev_cfg = json.load(f)

    bev_mapper = BEVMapper(
        H_pixel_to_world=bev_cfg["H_pixel_to_world"],
        bev_bounds=bev_cfg["bev_bounds"],
        bev_resolution=bev_cfg["bev_resolution"],
    )

    # Read the real FPS from the video instead of assuming 25.0 --
    # a wrong FPS silently scales every PET time-gap computation.
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    traj_logger = TrajectoryLogger(fps=fps)

    pipe = ContactPointPipeline(model_name="yolo26n-seg.pt")
    df = pipe.run(
        video_path=str(video_path),
        conf=conf,
        max_frames=max_frames,
    )

    if df.empty:
        intervals: List[IntervalType] = []
        pet_events: List[PETEventType] = []
        pet_summary = summarize_pet(pet_events)
        traj_stats = traj_logger.get_stats()
        return YOLO26SegGridPETResult(
            fps=float(fps),
            frame_count=0,
            det_count_total=0,
            intervals=intervals,
            pet_events=pet_events,
            pet_summary=pet_summary,
            traj_stats=traj_stats,
            processing_time_seconds=time.time() - start_time,
        )

    df = df.sort_values(["frame_idx", "track_id"]).reset_index(drop=True)

    for _, row in df.iterrows():
        # Respect the road-mask validity check computed upstream in
        # ContactPointPipeline -- previously this was silently ignored,
        # letting off-road detections get projected anyway.
        if "valid_road_mask" in row and not bool(row["valid_road_mask"]):
            continue

        frame_idx = int(row["frame_idx"])
        track_id = int(row["track_id"])
        cx = float(row["pixel_x"])
        cy = float(row["pixel_y"])

        cell_id = grid.get_cell_from_pixels(cx, cy)
        if isinstance(cell_id, str) and cell_id.upper() == "OUT_OF_BOUNDS":
            continue

        wx = row["world_x"]
        wy = row["world_y"]

        if pd.isna(wx) or pd.isna(wy):
            world_xy = bev_mapper.pixel_to_world((cx, cy))
            if world_xy is not None:
                wx, wy = float(world_xy[0]), float(world_xy[1])
            else:
                wx, wy = None, None
        else:
            wx, wy = float(wx), float(wy)

        traj_logger.log(
            track_id=track_id,
            frame_idx=frame_idx,
            cell_id=cell_id,
            world_x=wx,
            world_y=wy,
        )

    intervals: List[IntervalType] = traj_logger.build_intervals()
    pet_events: List[PETEventType] = compute_pet(
        intervals,
        pet_threshold=pet_threshold,
    )
    pet_summary = summarize_pet(pet_events)
    traj_stats = traj_logger.get_stats()

    frame_count = int(df["frame_idx"].nunique())
    det_count_total = int(len(df))

    return YOLO26SegGridPETResult(
        fps=float(fps),
        frame_count=frame_count,
        det_count_total=det_count_total,
        intervals=intervals,
        pet_events=pet_events,
        pet_summary=pet_summary,
        traj_stats=traj_stats,
        processing_time_seconds=time.time() - start_time,
    )
