from __future__ import annotations
import matplotlib.pyplot as plt
from src.utils.interactive import show_image, ask_user
from src.analysis.grid_trajectory.spatial_grid import SpatialGrid
from src.bev.bev_mapper import BEVMapper
from tqdm import tqdm
import sys
import json


from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Tuple

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import torch


UVH_DISPLAY_MAP = {
    "Three-wheeler": "auto",
    "Two-wheeler": "bike",
    "Hatchback": "car",
    "Sedan": "car",
    "SUV": "car",
    "MUV": "car",
    "Van": "car",
    "Truck": "truck",
    "LCV": "truck",
    "Bus": "bus",
    "Mini-bus": "bus",
    "tempo-traveller": "bus",
}

CLASS_NAME_TO_ID = {
    "pedestrian": 0,
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "bike": 3,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7,
    "auto": 8,
}


@dataclass
class TrackPoint:
    frame: int
    x: float
    y: float
    cls_id: int
    cls_name: str
    conf: float


def _segment_intersection(p1, p2, q1, q2):
    p = np.array(p1, dtype=float)
    r = np.array(p2, dtype=float) - p
    q = np.array(q1, dtype=float)
    s = np.array(q2, dtype=float) - q

    rxs = np.cross(r, s)
    q_p = q - p
    qpxr = np.cross(q_p, r)

    if abs(rxs) < 1e-9 and abs(qpxr) < 1e-9:
        return None
    if abs(rxs) < 1e-9 and abs(qpxr) >= 1e-9:
        return None

    t = np.cross(q_p, s) / rxs
    u = np.cross(q_p, r) / rxs

    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        inter = p + t * r
        return float(inter[0]), float(inter[1])
    return None


def _point_in_square(px, py, cx, cy, half_size):
    return (cx - half_size) <= px <= (cx + half_size) and (cy - half_size) <= py <= (cy + half_size)


def _entry_exit_frames(points: List[TrackPoint], cx: float, cy: float, half_size: float):
    inside_frames = [pt.frame for pt in points if _point_in_square(pt.x, pt.y, cx, cy, half_size)]
    if not inside_frames:
        return None
    return min(inside_frames), max(inside_frames)


def _pair_conflict_point(track_a: List[TrackPoint], track_b: List[TrackPoint]):
    for i in range(len(track_a) - 1):
        p1 = (track_a[i].x, track_a[i].y)
        p2 = (track_a[i + 1].x, track_a[i + 1].y)
        for j in range(len(track_b) - 1):
            q1 = (track_b[j].x, track_b[j].y)
            q2 = (track_b[j + 1].x, track_b[j + 1].y)
            inter = _segment_intersection(p1, p2, q1, q2)
            if inter is not None:
                return inter
    return None


def _box_intersection(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float((x2 - x1) * (y2 - y1))


def _box_area(box: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _overlap_over_person(person_box: Tuple[float, float, float, float], other_box: Tuple[float, float, float, float]) -> float:
    inter = _box_intersection(person_box, other_box)
    area = _box_area(person_box)
    return 0.0 if area <= 0 else inter / area


def run_uvh_coco_fused_grid_pet(
    video_path: str,
    bev_config_path: str,
    grid_config_path: str,
    uvh_model_path: str,
    coco_person_model_path: str,
    output_csv_path: str,
    pet_threshold: float = 2.0,
    max_frames: int | None = None,
    imgsz: int = 1280,
    uvh_conf: float = 0.20,
    coco_person_conf: float = 0.20,
    person_suppress_overlap: float = 0.35,
    show_progress: bool = True, interactive: bool = False,
    device: str = "auto",
    backend: str = "auto",
) -> Dict[str, Any]:
    video_path = str(Path(video_path).resolve())
    uvh_model_path = str(Path(uvh_model_path).resolve())
    coco_person_model_path = str(Path(coco_person_model_path).resolve())
    output_csv_path = str(Path(output_csv_path).resolve())

    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Grid + BEV mappers for detailed PET output
    spatial_grid = None
    bev_mapper = None
    try:
        spatial_grid = SpatialGrid(grid_config_path)
    except Exception:
        spatial_grid = None
    try:
        import json as _json
        with open(bev_config_path) as f:
            bev_cfg = _json.load(f)
        H = np.array(bev_cfg["H_pixel_to_world"], dtype=np.float32)
        bounds = bev_cfg
        bev_res = bev_cfg.get("bev_resolution", bev_cfg.get("resolution", [1000, 800]))
        bev_mapper = BEVMapper(H, bounds, bev_res)
    except Exception:
        bev_mapper = None

    # Determine OpenVINO model directories
    def _openvino_dir(model_pt_path: str) -> Path:
        p = Path(model_pt_path)
        return p.with_name(p.stem + "_openvino_model")

    use_openvino = False
    if backend == "auto":
        if torch.cuda.is_available():
            device = "cuda:0" if device == "auto" else device
        else:
            try:
                import openvino
                ov_uvh_dir = _openvino_dir(uvh_model_path)
                ov_coco_dir = _openvino_dir(coco_person_model_path)
                if ov_uvh_dir.exists() and ov_coco_dir.exists():
                    use_openvino = True
            except Exception:
                pass
    elif backend == "openvino":
        use_openvino = True

    if use_openvino:
        uvh_model = YOLO(str(_openvino_dir(uvh_model_path)), task="detect")
        coco_model = YOLO(str(_openvino_dir(coco_person_model_path)), task="detect")
    else:
        uvh_model = YOLO(uvh_model_path)
        coco_model = YOLO(coco_person_model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    detection_rows = []
    tracks: Dict[int, List[TrackPoint]] = {}

    frame_idx = 0

    uvh_results = uvh_model.track(
        source=video_path,
        stream=True,
        persist=True,
        tracker="bytetrack.yaml",
        conf=uvh_conf,
        imgsz=imgsz,
        verbose=False,
        device=device,
    )

    coco_results = coco_model.track(
        source=video_path,
        stream=True,
        persist=True,
        tracker="bytetrack.yaml",
        conf=coco_person_conf,
        imgsz=imgsz,
        verbose=False,
        device=device,
        classes=[0],
    )

    total_iters = total_frames if max_frames is None else min(total_frames, max_frames)
    pbar = tqdm(total=total_iters, desc="Processing frames", unit="frame", disable=not show_progress)
    for uvh_r, coco_r in zip(uvh_results, coco_results):
        if max_frames is not None and frame_idx >= max_frames:
            break

        uvh_boxes_for_suppression = []

        uvh_boxes = uvh_r.boxes
        if uvh_boxes is not None and uvh_boxes.xyxy is not None and len(uvh_boxes) > 0:
            xyxy = uvh_boxes.xyxy.cpu().numpy()
            clss = uvh_boxes.cls.cpu().numpy().astype(int) if uvh_boxes.cls is not None else np.array([], dtype=int)
            confs = uvh_boxes.conf.cpu().numpy() if uvh_boxes.conf is not None else np.array([], dtype=float)

            if hasattr(uvh_boxes, "id") and uvh_boxes.id is not None:
                ids = uvh_boxes.id.cpu().numpy().astype(int)
            else:
                ids = np.arange(len(xyxy), dtype=int) + frame_idx * 1000

            for i, box in enumerate(xyxy):
                raw_name = uvh_r.names[int(clss[i])]
                mapped_name = UVH_DISPLAY_MAP.get(raw_name)
                if mapped_name is None:
                    continue

                x1, y1, x2, y2 = map(float, box.tolist())
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                track_id = int(ids[i])
                score = float(confs[i]) if i < len(confs) else 0.0
                cls_id = int(CLASS_NAME_TO_ID.get(mapped_name, 99))

                uvh_box = (x1, y1, x2, y2)
                uvh_boxes_for_suppression.append(uvh_box)

                detection_rows.append({
                    "frame": frame_idx,
                    "track_id": track_id,
                    "class_id": cls_id,
                    "class_name": mapped_name,
                    "conf": score,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "cx": cx,
                    "cy": cy,
                    "source": "uvh26",
                })

                tracks.setdefault(track_id, []).append(
                    TrackPoint(
                        frame=frame_idx,
                        x=cx,
                        y=cy,
                        cls_id=cls_id,
                        cls_name=mapped_name,
                        conf=score,
                    )
                )

        coco_boxes = coco_r.boxes
        if coco_boxes is not None and coco_boxes.xyxy is not None and len(coco_boxes) > 0:
            xyxy = coco_boxes.xyxy.cpu().numpy()
            confs = coco_boxes.conf.cpu().numpy() if coco_boxes.conf is not None else np.array([], dtype=float)

            if hasattr(coco_boxes, "id") and coco_boxes.id is not None:
                ids = coco_boxes.id.cpu().numpy().astype(int)
            else:
                ids = np.arange(len(xyxy), dtype=int) + frame_idx * 100000 + 50000

            for i, box in enumerate(xyxy):
                x1, y1, x2, y2 = map(float, box.tolist())
                person_box = (x1, y1, x2, y2)

                covered = any(
                    _overlap_over_person(person_box, veh_box) >= person_suppress_overlap
                    for veh_box in uvh_boxes_for_suppression
                )
                if covered:
                    continue

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                track_id = int(ids[i])
                score = float(confs[i]) if i < len(confs) else 0.0
                cls_id = 0
                cls_name = "pedestrian"

                detection_rows.append({
                    "frame": frame_idx,
                    "track_id": track_id,
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "conf": score,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "cx": cx,
                    "cy": cy,
                    "source": "coco_person",
                })

                tracks.setdefault(track_id, []).append(
                    TrackPoint(
                        frame=frame_idx,
                        x=cx,
                        y=cy,
                        cls_id=cls_id,
                        cls_name=cls_name,
                        conf=score,
                    )
                )

        if show_progress and frame_idx % 25 == 0:
            pass
        frame_idx += 1
        pbar.update(1)

        # === INTERACTIVE PAUSE ===
        if interactive and frame_idx % 20 == 0:
            print("\n" + "="*50)
            print(f"🔄 Processed {frame_idx} frames.")
            try:
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    from IPython.display import display
                    fig = plt.figure()
                    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    plt.title(f"Frame {frame_idx}")
                    plt.axis('off')
                    display(fig)
                    plt.close(fig)
            except Exception as e:
                print(f"⚠️ Could not show frame: {e}")

            print("Press Enter to continue, or type 'stop' and press Enter to abort.")
            sys.stdout.flush()
            user_input = input().strip().lower()
            if user_input == 'stop':
                print("⏹️ Stopped by user.")
                stop_processing = True
                break

    pbar.close()
    print(f"[UVH-COCO] backend={'openvino' if use_openvino else 'pytorch'} device={device}")
    print(f"[UVH-COCO] ✅ Frame processing finished ({frame_idx} frames). Initializing track indexing...", flush=True)

    det_df = pd.DataFrame(detection_rows)
    out_path = Path(output_csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    detections_csv = str(out_path.with_name(out_path.stem + "_detections.csv"))
    det_df.to_csv(detections_csv, index=False)

    pet_events = []
    event_id = 1
    conflict_half_size = 20.0

    valid_tracks = {tid: pts for tid, pts in tracks.items() if len(pts) >= 3}
    track_items = list(valid_tracks.items())

    # Precompute track metadata for fast temporal/spatial pruning
    track_meta = {}
    for tid, pts in valid_tracks.items():
        frames = [p.frame for p in pts]
        xs = [p.x for p in pts]
        ys = [p.y for p in pts]
        track_meta[tid] = {
            "frames": frames,
            "min_frame": min(frames),
            "max_frame": max(frames),
            "min_x": min(xs),
            "max_x": max(xs),
            "min_y": min(ys),
            "max_y": max(ys),
        }

    max_frames_diff = int(pet_threshold * fps) + 5  # temporal padding
    spatial_pad = 50.0  # pixel padding for bounding box check

    for i in range(len(track_items)):
        track_a_id, pts_a = track_items[i]
        meta_a = track_meta[track_a_id]
        for j in range(i + 1, len(track_items)):
            track_b_id, pts_b = track_items[j]
            meta_b = track_meta[track_b_id]

            # Temporal pruning: skip if track lifetimes are too far apart
            if (meta_a["min_frame"] > meta_b["max_frame"] + max_frames_diff or
                meta_b["min_frame"] > meta_a["max_frame"] + max_frames_diff):
                continue

            # Spatial pruning: skip if bounding boxes do not overlap (with padding)
            if (meta_a["max_x"] < meta_b["min_x"] - spatial_pad or
                meta_a["min_x"] > meta_b["max_x"] + spatial_pad or
                meta_a["max_y"] < meta_b["min_y"] - spatial_pad or
                meta_a["min_y"] > meta_b["max_y"] + spatial_pad):
                continue

            inter = _pair_conflict_point(pts_a, pts_b)
            if inter is None:
                continue

            cx, cy = inter
            a_window = _entry_exit_frames(pts_a, cx, cy, conflict_half_size)
            b_window = _entry_exit_frames(pts_b, cx, cy, conflict_half_size)
            if a_window is None or b_window is None:
                continue

            a_entry, a_exit = a_window
            b_entry, b_exit = b_window

            if a_exit <= b_entry:
                pet = (b_entry - a_exit) / fps
                first_id, second_id = track_a_id, track_b_id
                frame_ref = b_entry
            elif b_exit <= a_entry:
                pet = (a_entry - b_exit) / fps
                first_id, second_id = track_b_id, track_a_id
                frame_ref = a_entry
            else:
                continue

            if pet <= pet_threshold:
                # Determine grid cell for conflict point
                grid_cell = spatial_grid.get_cell_from_pixels(cx, cy) if spatial_grid else "UNKNOWN"

                pet_events.append({
                    "event_id": event_id,
                    "pet": float(pet),
                    "frame": int(frame_ref),
                    "track_a": int(first_id),
                    "track_b": int(second_id),
                    "conflict_type": "image_intersection",
                    "grid_cell": grid_cell,
                    "entry_frame_a": int(a_entry),
                    "exit_frame_a": int(a_exit),
                    "entry_frame_b": int(b_entry),
                    "exit_frame_b": int(b_exit),
                    "world_traj_i": f"track_{first_id}",
                    "world_traj_j": f"track_{second_id}",
                    "traj_a_json": _track_to_json(pts_a if first_id == track_a_id else pts_b, bev_mapper),
                    "traj_b_json": _track_to_json(pts_b if first_id == track_a_id else pts_a, bev_mapper),
                })
                event_id += 1

    pet_df = pd.DataFrame(
        pet_events,
        columns=[
            "event_id",
            "pet",
            "frame",
            "track_a",
            "track_b",
            "conflict_type",
            "grid_cell",
            "entry_frame_a",
            "exit_frame_a",
            "entry_frame_b",
            "exit_frame_b",
            "world_traj_i",
            "world_traj_j",
            "traj_a_json",
            "traj_b_json",
        ],
    )
    pet_df.to_csv(output_csv_path, index=False)

    print(f"[UVH-COCO] detections_csv={detections_csv} rows={len(det_df)}")
    print(f"[UVH-COCO] pet_csv={output_csv_path} rows={len(pet_df)}")
    print(f"[UVH-COCO] fps={fps:.3f} valid_tracks={len(valid_tracks)}")

    return {
        "detections_csv": detections_csv,
        "pet_csv": output_csv_path,
        "pet_events": pet_events,
        "fps": fps,
        "num_tracks": len(valid_tracks),
    }



def _track_to_json(points: List[TrackPoint], bev_mapper=None) -> str:
    """Convert a list of TrackPoint to a JSON string with pixel and world coords."""
    rows = []
    for pt in points:
        row = {
            "frame": int(pt.frame),
            "x_pixel": float(pt.x),
            "y_pixel": float(pt.y),
            "world_x": None,
            "world_y": None,
        }
        if bev_mapper is not None:
            world = bev_mapper.pixel_to_world((pt.x, pt.y))
            if world is not None:
                row["world_x"] = float(world[0])
                row["world_y"] = float(world[1])
        rows.append(row)
    return json.dumps(rows)

def _can_intersect_temporal(track_a, track_b, fps=25.0, max_pet=2.0):
    """O(1) temporal check: Skip tracks whose lifetime windows do not overlap within max_pet seconds."""
    max_frames_diff = int(max_pet * fps) + 5
    min_a, max_a = track_a["frames"][0], track_a["frames"][-1]
    min_b, max_b = track_b["frames"][0], track_b["frames"][-1]
    
    if min_b > (max_a + max_frames_diff) or min_a > (max_b + max_frames_diff):
        return False
    return True
