import os, cv2, numpy as np, json
from ultralytics.models.sam import SAM3VideoSemanticPredictor
from grid_trajectory.spatial_grid import SpatialGrid
from grid_trajectory.pet_grid import TrajectoryLogger, compute_pet
from bev_mapper import BEVMapper

def run_sam3_grid_pet(
    project_root="/content/drive/MyDrive/shared_pipeline/4D_tracking_project",
    video_rel_path="videos/traffic_video_50frames.mp4",
    sam3_rel_path="sam3.pt",
    grid_rel_path="configs/GITI_grid_config.json",
    bev_rel_path="configs/bev_config.json",
    output_name="sam3_grid_pet_run",
    conf=0.25,
    pet_threshold=2.0,
):
    video_path  = os.path.join(project_root, video_rel_path)
    sam3_path   = os.path.join(project_root, sam3_rel_path)
    grid_config = os.path.join(project_root, grid_rel_path)
    bev_config  = os.path.join(project_root, bev_rel_path)

    output_root = os.path.join(project_root, "outputs")
    os.makedirs(output_root, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    ret, frame0 = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Could not read first frame for grid/BEV initialization.")

    h0, w0 = frame0.shape[:2]
    grid = SpatialGrid(grid_config)

    with open(bev_config, "r") as f:
        bev_cfg = json.load(f)
    bev_mapper = BEVMapper(
        H_pixel_to_world=bev_cfg["H_pixel_to_world"],
        bev_bounds=bev_cfg["bev_bounds"],
        bev_resolution=bev_cfg["bev_resolution"],
    )

    traj_logger = TrajectoryLogger(fps=fps)

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

    MAX_FRAMES = 300  # temporary frame cap for debugging
    for frame_idx, res in enumerate(results_gen):
        if frame_idx >= MAX_FRAMES:
            break
        frame_count += 1

        boxes = getattr(res, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            continue

        xyxy = boxes.xyxy.detach().cpu().numpy()
        if xyxy.size == 0:
            continue

        track_ids = getattr(boxes, "id", None)
        if track_ids is not None:
            track_ids = track_ids.detach().cpu().numpy()
        else:
            track_ids = np.arange(len(xyxy))

        h, w = res.orig_img.shape[:2]
        det_count_total += len(xyxy)

        for k, box in enumerate(xyxy):
            x1, y1, x2, y2 = box.astype(int)
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

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

    intervals = traj_logger.build_intervals()
    pet_events = compute_pet(intervals, pet_threshold=pet_threshold)

    return dict(
        fps=fps,
        frame_count=frame_count,
        det_count_total=det_count_total,
        intervals=intervals,
        pet_events=pet_events,
    )
