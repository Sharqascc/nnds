import os
import cv2
import numpy as np
import json

from ultralytics.models.sam import SAM3VideoSemanticPredictor
from grid_trajectory.spatial_grid import SpatialGrid
from grid_trajectory.pet_grid import TrajectoryLogger, compute_pet
from bev_mapper import BEVMapper

# Visualization config: per-class BGR colors
CLASS_COLORS = {
    "car": (0, 0, 0),               # black
    "bus": (255, 0, 0),             # blue
    "motorcycle": (0, 255, 0),      # green
    "auto rickshaw": (0, 255, 255), # yellow
    "pedestrian": (0, 0, 255),      # red
    "bicycle": (255, 255, 0),       # cyan
}


def run_sam3_grid_pet(
    project_root="/content/drive/MyDrive/shared_pipeline/4D_tracking_project",
    video_rel_path="videos/traffic_video_50frames.mp4",
    sam3_rel_path="sam3.pt",
    grid_rel_path="configs/GITI_grid_config.json",
    bev_rel_path="configs/bev_config.json",
    output_name="sam3_grid_pet_run",
    conf=0.25,
    pet_threshold=2.0,
    max_frames: int | None = None,
    debug_video_rel_path: str | None = None,
):
    """
    SAM3 + grid + BEV + PET pipeline with optional per-class colored debug video.
    Returns a dict with fps, frame_count, det_count_total, intervals, pet_events.
    """

    video_path  = os.path.join(project_root, video_rel_path)
    sam3_path   = os.path.join(project_root, sam3_rel_path)
    grid_config = os.path.join(project_root, grid_rel_path)
    bev_config  = os.path.join(project_root, bev_rel_path)

    output_root = os.path.join(project_root, "outputs")
    os.makedirs(output_root, exist_ok=True)

    # Open video once for FPS/size and to support VideoWriter
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    ret, frame0 = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Could not read first frame for grid/BEV initialization.")

    h0, w0 = frame0.shape[:2]

    # Optional debug video writer
    writer = None
    if debug_video_rel_path is not None:
        debug_video_path = os.path.join(project_root, debug_video_rel_path)
        os.makedirs(os.path.dirname(debug_video_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(debug_video_path, fourcc, fps, (w0, h0))

    # Grid and BEV setup
    grid = SpatialGrid(grid_config)

    with open(bev_config, "r") as f:
        bev_cfg = json.load(f)
    bev_mapper = BEVMapper(
        H_pixel_to_world=bev_cfg["H_pixel_to_world"],
        bev_bounds=bev_cfg["bev_bounds"],
        bev_resolution=bev_cfg["bev_resolution"],
    )

    traj_logger = TrajectoryLogger(fps=fps)

    # SAM3 predictor setup
    overrides = dict(
        conf=conf,
        task="segment",
        mode="predict",
        imgsz=640,
        model=sam3_path,
        half=True,
        save=False,
        project=output_root,
        name=output_name,
    )
    predictor = SAM3VideoSemanticPredictor(overrides=overrides)
    concepts  = ["car", "bus", "motorcycle", "auto rickshaw", "pedestrian", "bicycle"]

    results_gen = predictor(source=video_path, text=concepts, stream=True)

    frame_count = 0
    det_count_total = 0

    for frame_idx, res in enumerate(results_gen):
        if max_frames is not None and frame_idx >= max_frames:
            break
        frame_count += 1

        boxes = getattr(res, "boxes", None)
        frame = res.orig_img.copy()

        if boxes is None or boxes.xyxy is None:
            if writer is not None:
                writer.write(frame)
            continue

        xyxy = boxes.xyxy.detach().cpu().numpy()
        if xyxy.size == 0:
            if writer is not None:
                writer.write(frame)
            continue

        track_ids = getattr(boxes, "id", None)
        if track_ids is not None:
            track_ids = track_ids.detach().cpu().numpy()
        else:
            track_ids = np.arange(len(xyxy))

        cls_indices = getattr(boxes, "cls", None)
        if cls_indices is not None:
            cls_indices = cls_indices.detach().cpu().numpy()
        else:
            cls_indices = np.zeros(len(xyxy), dtype=int)

        h, w = frame.shape[:2]
        det_count_total += len(xyxy)

        for k, box in enumerate(xyxy):
            x1, y1, x2, y2 = box.astype(int)
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            # determine class name and color
            cls_idx = int(cls_indices[k]) if k < len(cls_indices) else 0
            cls_name = concepts[cls_idx] if cls_idx < len(concepts) else "unknown"
            color = CLASS_COLORS.get(cls_name, (0, 255, 0))  # default green

            # Draw detection + track id + class name in per-class color
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} ID {int(track_ids[k])}"
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

            # bottom-center point for grid / BEV
            cx = (x1 + x2) / 2.0
            cy = float(y2)

            cell_id = grid.get_cell_from_pixels(cx, cy)
            if isinstance(cell_id, str) and cell_id.upper() == "OUT_OF_BOUNDS":
                continue

            world_xy = bev_mapper.pixel_to_world((cx, cy))
            if world_xy is not None:
                wx, wy = world_xy
            else:
                wx, wy = None, None

            traj_logger.log(
                track_id=track_ids[k],
                frame_idx=frame_idx,
                cell_id=cell_id,
                world_x=wx,
                world_y=wy,
            )

        if writer is not None:
            writer.write(frame)

    cap.release()
    if writer is not None:
        writer.release()

    intervals = traj_logger.build_intervals()
    pet_events = compute_pet(intervals, pet_threshold=pet_threshold)

    return dict(
        fps=fps,
        frame_count=frame_count,
        det_count_total=det_count_total,
        intervals=intervals,
        pet_events=pet_events,
    )
