from __future__ import annotations

"""
SAM3 + grid + BEV pipeline to compute PET events from raw video.

This module:
- Runs SAM3VideoSemanticPredictor on a video
- Maps detections into a SpatialGrid
- Logs trajectories with TrajectoryLogger
- Computes PET events and summary stats using grid_trajectory.pet_grid
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import logging
import time
import json
from collections import defaultdict

import cv2
import numpy as np

from grid_trajectory.spatial_grid import SpatialGrid
from grid_trajectory.pet_grid import (
    TrajectoryLogger,
    compute_pet,
    summarize_pet,
    PETEventType,
    IntervalType,
    PETSummaryType,
)
from core.bev_mapper import BEVMapper


LoggerType = Union[logging.Logger, Any]


def _get_sam3_predictor_cls():
    try:
        from ultralytics.models.sam import SAM3VideoSemanticPredictor
    except ImportError as exc:
        raise ImportError(
            "run_sam3_grid_pet requires the optional dependency 'ultralytics'. "
            "Install it before using the SAM3 pipeline."
        ) from exc
    return SAM3VideoSemanticPredictor



@dataclass
class SAM3GridPETResult:
    fps: float
    frame_count: int
    det_count_total: int
    intervals: List[IntervalType]
    pet_events: List[PETEventType]
    pet_summary: PETSummaryType
    traj_stats: Dict[str, Any]
    processing_time_seconds: float = 0.0

    def print_summary(self) -> None:
        """Print formatted summary to stdout."""
        print("\n" + "=" * 60)
        print("📊 SAM3+Grid+PET Results")
        print("=" * 60)
        print(f"Frames processed:  {self.frame_count:,}")
        print(f"Total detections:  {self.det_count_total:,}")
        print(f"PET events:        {len(self.pet_events):,}")
        if self.processing_time_seconds > 0:
            print(f"Processing time:   {self.processing_time_seconds:.1f}s")

        ps = self.pet_summary
        print("\nPET Statistics:")
        if ps.mean_pet is not None:
            print(f"  Mean:    {ps.mean_pet:.3f}s")
        if ps.p50 is not None:
            print(f"  Median:  {ps.p50:.3f}s")
        print(f"  Critical events: {ps.n_critical}")
        print("=" * 60)


def _validate_conf(conf: float) -> None:
    if not (0.0 <= conf <= 1.0):
        raise ValueError(f"conf must be in [0, 1], got {conf}")


def _validate_max_frames(max_frames: Optional[int]) -> None:
    if max_frames is not None and max_frames <= 0:
        raise ValueError(f"max_frames must be positive, got {max_frames}")


def _validate_frame_stride(frame_stride: int) -> None:
    if frame_stride <= 0:
        raise ValueError(f"frame_stride must be positive, got {frame_stride}")


def _validate_bev_config(cfg: Dict[str, Any]) -> None:
    required_keys = {"H_pixel_to_world", "bev_bounds", "bev_resolution"}
    missing = required_keys - set(cfg.keys())
    if missing:
        raise KeyError(f"BEV config missing keys: {sorted(missing)}")


def run_sam3_grid_pet(
    project_root: str | Path,
    video_rel_path: str = "videos/traffic_video_50frames.mp4",
    sam3_rel_path: str = "sam3.pt",
    grid_rel_path: str = "configs/GITI_grid_config.json",
    bev_rel_path: str = "configs/bev_config.json",
    output_name: str = "sam3_grid_pet_run",
    conf: float = 0.1,
    pet_threshold: float = 2.0,
    max_frames: Optional[int] = None,
    debug_video_rel_path: Optional[str] = None,
    # Configurable SAM3 knobs
    imgsz: int = 640,
    half: bool = True,
    concepts: Optional[Sequence[str]] = None,
    # Frame stride for faster debugging
    frame_stride: int = 1,
    # Logging / progress
    show_progress: bool = True,
    logger: Optional[LoggerType] = None,
) -> SAM3GridPETResult:
    """
    Run SAM3 segmentation + grid mapping + PET extraction on a single video.

    Args:
        project_root: Root directory of the NNDS project.
        video_rel_path: Video path relative to project_root.
        sam3_rel_path: SAM3 model path relative to project_root.
        grid_rel_path: Grid config JSON relative to project_root.
        bev_rel_path: BEV config JSON relative to project_root.
        output_name: Name for SAM3 predictor output folder.
        conf: Detection confidence threshold for SAM3 (0–1).
        pet_threshold: Max PET (seconds) to record (must be > 0).
        max_frames: Optional max number of frames to process.
        debug_video_rel_path: Optional relative path for debug MP4 output.
        imgsz: SAM3 input size.
        half: Whether to use half precision (if supported).
        concepts: Optional list of text prompts for SAM3.
        frame_stride: Process every k-th frame for faster debugging.
        show_progress: Print simple progress updates.
        logger: Optional logger implementing .info().

    Returns:
        SAM3GridPETResult with raw PET events, summary stats, and timing.
    """
    start_time = time.time()

    root = Path(project_root)
    video_path = root / video_rel_path
    sam3_path = root / sam3_rel_path
    grid_config = root / grid_rel_path
    bev_config = root / bev_rel_path

    # Basic validation
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not sam3_path.is_file():
        raise FileNotFoundError(f"SAM3 weights not found: {sam3_path}")
    if not grid_config.is_file():
        raise FileNotFoundError(f"Grid config not found: {grid_config}")
    if not bev_config.is_file():
        raise FileNotFoundError(f"BEV config not found: {bev_config}")
    if pet_threshold <= 0:
        raise ValueError(f"pet_threshold must be positive, got {pet_threshold}")
    _validate_conf(conf)
    _validate_max_frames(max_frames)
    _validate_frame_stride(frame_stride)

    output_root = root / "outputs"
    output_root.mkdir(parents=True, exist_ok=True)

    # Open video once for FPS/size and to support VideoWriter
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    ret, frame0 = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Could not read first frame for grid/BEV initialization.")

    h0, w0 = frame0.shape[:2]

    # Optional debug video writer
    writer: Optional[cv2.VideoWriter] = None
    combined_width = w0 + 420
    combined_height = h0
    if debug_video_rel_path is not None:
        debug_video_path = root / debug_video_rel_path
        debug_video_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(debug_video_path), fourcc, fps, (combined_width, combined_height))

    # Grid and BEV mapper
    grid = SpatialGrid(str(grid_config))

    with bev_config.open("r") as f:
        bev_cfg = json.load(f)
    _validate_bev_config(bev_cfg)

    bev_mapper = BEVMapper(
        H_pixel_to_world=bev_cfg["H_pixel_to_world"],
        bev_bounds=bev_cfg["bev_bounds"],
        bev_resolution=bev_cfg["bev_resolution"],
    )

    traj_logger = TrajectoryLogger(fps=fps)

    track_history: Dict[int, List[tuple[int, int]]] = defaultdict(list)
    world_history: Dict[int, List[tuple[float, float]]] = defaultdict(list)

    bev_w, bev_h = bev_cfg["bev_resolution"]

    def _world_to_bev_pixel(wx: float, wy: float) -> tuple[int, int]:
        x_min = bev_cfg["bev_bounds"]["x_min"]
        x_max = bev_cfg["bev_bounds"]["x_max"]
        y_min = bev_cfg["bev_bounds"]["y_min"]
        y_max = bev_cfg["bev_bounds"]["y_max"]

        if x_max == x_min or y_max == y_min:
            return 0, 0

        u = int((wx - x_min) / (x_max - x_min) * (bev_w - 1))
        v = int((1.0 - (wy - y_min) / (y_max - y_min)) * (bev_h - 1))
        u = max(0, min(u, bev_w - 1))
        v = max(0, min(v, bev_h - 1))
        return u, v

    def _make_bev_canvas() -> np.ndarray:
        canvas = np.full((bev_h, bev_w, 3), 20, dtype=np.uint8)
        cv2.rectangle(canvas, (0, 0), (bev_w - 1, bev_h - 1), (120, 120, 120), 2)
        cv2.putText(
            canvas,
            "Bird's Eye View",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return canvas

    # SAM3 overrides
    overrides = dict(
        conf=conf,
        task="segment",
        mode="predict",
        imgsz=imgsz,
        model=str(sam3_path),
        half=half,
        save=False,
        project=str(output_root),
        name=output_name,
    )
    SAM3VideoSemanticPredictor = _get_sam3_predictor_cls()
    predictor = SAM3VideoSemanticPredictor(overrides=overrides)

    # Default concepts if none provided
    if concepts is None:
        concepts = [
            "car",
            "bus",
            "motorcycle",
            "auto rickshaw",
            "pedestrian",
            "bicycle",
        ]

    results_gen = predictor(source=str(video_path), text=list(concepts), stream=True)

    frame_count = 0
    det_count_total = 0

    def _log(msg: str) -> None:
        if logger is not None:
            logger.info(msg)
        elif show_progress:
            print(msg)

    try:
        for frame_idx, res in enumerate(results_gen):
            if max_frames is not None and frame_idx >= max_frames:
                break

            # Frame subsampling for faster debug runs
            if frame_idx % frame_stride != 0:
                continue

            frame_count += 1

            # Some SAM3 frames may be corrupted or missing attributes
            try:
                boxes = getattr(res, "boxes", None)
                frame = res.orig_img.copy()
            except Exception as e:  # noqa: BLE001
                _log(f"⚠️ Skipping frame {frame_idx} due to SAM3 error: {e}")
                continue

            if boxes is None or getattr(boxes, "xyxy", None) is None:
                if writer is not None:
                    writer.write(frame)
                continue

            xyxy = boxes.xyxy.detach().cpu().numpy()
            if xyxy.size == 0:
                if writer is not None:
                    writer.write(frame)
                continue

            track_ids_attr = getattr(boxes, "id", None)
            if track_ids_attr is not None:
                track_ids = track_ids_attr.detach().cpu().numpy()
                ids_are_synthetic = False
            else:
                # SAM3 did not return persistent IDs for this frame.
                # Synthetic per-frame indices have NO cross-frame identity
                # meaning and would corrupt TrajectoryLogger intervals if
                # logged as if they were real tracks. Warn loudly and skip.
                _log(
                    f"WARNING: frame {frame_idx} has no persistent track IDs "
                    f"from SAM3 (boxes.id is None). Skipping {len(xyxy)} detection(s) "
                    f"this frame rather than fabricating fake track identity."
                )
                track_ids = np.arange(len(xyxy), dtype=int)
                ids_are_synthetic = True

            h, w = frame.shape[:2]
            det_count_total += len(xyxy)

            draw_debug = writer is not None
            bev_canvas = _make_bev_canvas()

            cls_attr = getattr(boxes, "cls", None)
            conf_attr = getattr(boxes, "conf", None)
            cls_ids = cls_attr.detach().cpu().numpy().astype(int) if cls_attr is not None else None
            confs = conf_attr.detach().cpu().numpy() if conf_attr is not None else None

            names = getattr(res, "names", {}) or {}

            def _cls_name_from_id(cls_id: int) -> str:
                if isinstance(names, dict):
                    return str(names.get(cls_id, f"class_{cls_id}"))
                if isinstance(names, (list, tuple)):
                    if 0 <= cls_id < len(names):
                        return str(names[cls_id])
                return f"class_{cls_id}"

            for k, box in enumerate(xyxy):
                if ids_are_synthetic:
                    # Do not log fabricated identity into PET trajectories
                    continue
                x1, y1, x2, y2 = box.astype(int)
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))

                track_id = int(track_ids[k])
                cls_name = "obj"
                if cls_ids is not None and k < len(cls_ids):
                    cls_name = _cls_name_from_id(int(cls_ids[k]))
                conf_txt = ""
                if confs is not None and k < len(confs):
                    conf_txt = f" {float(confs[k]):.2f}"

                if draw_debug:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{cls_name} ID:{track_id}{conf_txt}"
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(y1 - 8, 18)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                cx = (x1 + x2) / 2.0
                cy = float(y2)

                if draw_debug:
                    cv2.circle(frame, (int(cx), int(cy)), 4, (0, 200, 255), -1)

                cell_id = grid.get_cell_from_pixels(cx, cy)
                if isinstance(cell_id, str) and cell_id.upper() == "OUT_OF_BOUNDS":
                    continue

                world_xy = bev_mapper.pixel_to_world((cx, cy))
                if world_xy is not None:
                    wx, wy = float(world_xy[0]), float(world_xy[1])
                    bx, by = _world_to_bev_pixel(wx, wy)

                    world_history[track_id].append((wx, wy))
                    track_history[track_id].append((bx, by))
                    if len(track_history[track_id]) > 30:
                        track_history[track_id].pop(0)
                    if len(world_history[track_id]) > 30:
                        world_history[track_id].pop(0)

                    if draw_debug:
                        pts = np.array(track_history[track_id], dtype=np.int32).reshape(-1, 1, 2)
                        if len(pts) > 1:
                            cv2.polylines(bev_canvas, [pts], False, (255, 180, 0), 2)
                        cv2.circle(bev_canvas, (bx, by), 6, (0, 220, 255), -1)
                        cv2.putText(
                            bev_canvas,
                            f"{cls_name}:{track_id}",
                            (min(bx + 8, bev_w - 120), max(by - 8, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
                else:
                    wx, wy = None, None

                traj_logger.log(
                    track_id=track_id,
                    frame_idx=frame_idx,
                    cell_id=cell_id,
                    world_x=wx,
                    world_y=wy,
                )

            if writer is not None and draw_debug:
                time_sec = frame_idx / fps if fps > 0 else 0.0
                cv2.rectangle(frame, (10, 10), (260, 95), (25, 25, 25), -1)
                cv2.rectangle(frame, (10, 10), (260, 95), (80, 80, 80), 1)
                cv2.putText(frame, f"Frame: {frame_idx}", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Time: {time_sec:.2f}s", (20, 62),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Detections: {len(xyxy)}", (20, 88),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

                bev_panel_w = 420
                bev_panel = np.full((h, bev_panel_w, 3), 30, dtype=np.uint8)
                inner_w = bev_panel_w - 20
                scale = min(inner_w / bev_w, (h - 20) / bev_h)
                resized = cv2.resize(bev_canvas, (int(bev_w * scale), int(bev_h * scale)))
                y0 = max((h - resized.shape[0]) // 2, 10)
                x0 = 10
                bev_panel[y0:y0+resized.shape[0], x0:x0+resized.shape[1]] = resized

                combined = np.hstack([frame, bev_panel])
                writer.write(combined)
            elif writer is not None:
                writer.write(frame)

            if show_progress and frame_idx % (50 * frame_stride) == 0:
                _log(
                    f"[SAM3+GRID] frame={frame_idx}, "
                    f"detections={len(xyxy)}, total_det={det_count_total}"
                )

    finally:
        cap.release()
        if writer is not None:
            writer.release()

    intervals: List[IntervalType] = traj_logger.build_intervals()
    pet_events: List[PETEventType] = compute_pet(
        intervals,
        pet_threshold=pet_threshold,
    )
    pet_summary = summarize_pet(pet_events)
    traj_stats = traj_logger.get_stats()

    processing_time = time.time() - start_time

    return SAM3GridPETResult(
        fps=float(fps),
        frame_count=frame_count,
        det_count_total=det_count_total,
        intervals=intervals,
        pet_events=pet_events,
        pet_summary=pet_summary,
        traj_stats=traj_stats,
        processing_time_seconds=processing_time,
    )


def run_sam3_grid_pet_batch(
    project_root: str | Path,
    video_paths: List[str],
    **kwargs: Any,
) -> List[SAM3GridPETResult]:
    """
    Run the SAM3+Grid+PET pipeline on multiple videos (sequentially).

    Args:
        project_root: Root of NNDS project.
        video_paths: List of video paths relative to project_root.
        **kwargs: Extra arguments forwarded to run_sam3_grid_pet.

    Returns:
        List of SAM3GridPETResult, one per video.
    """
    root = Path(project_root)
    results: List[SAM3GridPETResult] = []

    for rel_path in video_paths:
        print(f"\nProcessing: {rel_path}")
        result = run_sam3_grid_pet(
            project_root=root,
            video_rel_path=rel_path,
            **kwargs,
        )
        results.append(result)

    return results
