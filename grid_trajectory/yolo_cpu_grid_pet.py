
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


ALLOWED_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
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


def run_yolo_cpu_grid_pet(
    video_path: str,
    weights_path: str,
    output_csv_path: str,
    max_frames: int | None = None,
    imgsz: int = 480,
    conf: float = 0.25,
    pet_threshold: float = 2.0,
) -> Dict[str, Any]:
    video_path = str(Path(video_path).resolve())
    weights_path = str(Path(weights_path).resolve())
    output_csv_path = str(Path(output_csv_path).resolve())

    model = YOLO(weights_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    cap.release()

    detection_rows = []
    tracks: Dict[int, List[TrackPoint]] = {}

    frame_idx = 0
    results = model.track(
        source=video_path,
        stream=True,
        persist=True,
        tracker="bytetrack.yaml",
        conf=conf,
        imgsz=imgsz,
        verbose=False,
        classes=list(ALLOWED_CLASSES.keys()),
        device="cpu",
    )

    for r in results:
        if max_frames is not None and frame_idx >= max_frames:
            break

        boxes = r.boxes
        if boxes is not None and boxes.xyxy is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.array([], dtype=int)
            confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.array([], dtype=float)

            if hasattr(boxes, "id") and boxes.id is not None:
                ids = boxes.id.cpu().numpy().astype(int)
            else:
                ids = np.arange(len(xyxy), dtype=int) + frame_idx * 1000

            for i, box in enumerate(xyxy):
                cls_id = int(clss[i])
                if cls_id not in ALLOWED_CLASSES:
                    continue

                x1, y1, x2, y2 = map(float, box.tolist())
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                track_id = int(ids[i])
                score = float(confs[i]) if i < len(confs) else 0.0
                cls_name = ALLOWED_CLASSES[cls_id]

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

        frame_idx += 1

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

    for i in range(len(track_items)):
        track_a_id, pts_a = track_items[i]
        for j in range(i + 1, len(track_items)):
            track_b_id, pts_b = track_items[j]

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
                pet_events.append({
                    "event_id": event_id,
                    "pet": float(pet),
                    "frame": int(frame_ref),
                    "track_a": int(first_id),
                    "track_b": int(second_id),
                    "conflict_type": "image_intersection",
                    "world_traj_i": f"track_{first_id}",
                    "world_traj_j": f"track_{second_id}",
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
            "world_traj_i",
            "world_traj_j",
        ],
    )
    pet_df.to_csv(output_csv_path, index=False)

    print(f"[YOLO-CPU] detections_csv={detections_csv} rows={len(det_df)}")
    print(f"[YOLO-CPU] pet_csv={output_csv_path} rows={len(pet_df)}")
    print(f"[YOLO-CPU] fps={fps:.3f} valid_tracks={len(valid_tracks)}")

    return {
        "detections_csv": detections_csv,
        "pet_csv": output_csv_path,
        "pet_events": pet_events,
        "fps": fps,
        "num_tracks": len(valid_tracks),
    }
