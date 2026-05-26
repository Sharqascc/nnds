
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


@dataclass
class TrackPoint:
    frame_idx: int
    track_id: int
    cls_id: int
    conf: float
    pixel_x: float
    pixel_y: float
    world_x: Optional[float]
    world_y: Optional[float]
    valid_road_mask: bool


class ContactPointPipeline:

def _project_pixel_to_world(self, pixel_x, pixel_y):
    """Project a single pixel point to world coordinates using loaded homography."""
    if getattr(self, "H", None) is None:
        raise RuntimeError("Homography matrix self.H is not loaded")
    import numpy as np
    pt = np.array([float(pixel_x), float(pixel_y), 1.0], dtype=float)
    out = self.H @ pt
    if not np.isfinite(out).all():
        raise RuntimeError(f"Non-finite homography output for point {(pixel_x, pixel_y)}: {out}")
    if abs(out[2]) < 1e-12:
        raise RuntimeError(f"Homography normalization term too small for point {(pixel_x, pixel_y)}: {out}")
    return float(out[0] / out[2]), float(out[1] / out[2])

    def __init__(
        self,
        model_name: str = "yolo26n-seg.pt",
        road_mask_path: Optional[str] = None,
        homography_path: Optional[str] = None,
        classes: Optional[List[int]] = None,
    ) -> None:
        self.model = YOLO(model_name)
        self.road_mask = None
        self.H = None
        self.classes = classes or [2, 3, 5, 7]

        if road_mask_path:
            mask = cv2.imread(str(road_mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Could not read road mask: {road_mask_path}")
            self.road_mask = (mask > 0).astype(np.uint8)

        if homography_path:
            H = np.load(str(homography_path))
            if H.shape != (3, 3):
                raise ValueError(f"Homography must be 3x3, got {H.shape}")
            self.H = H.astype(float)

    @staticmethod
    def _box_bottom_center(box_xyxy: Optional[np.ndarray]) -> Tuple[float, float]:
        if box_xyxy is None:
            raise ValueError("No bbox available for fallback")
        x1, y1, x2, y2 = map(float, box_xyxy)
        return (x1 + x2) / 2.0, y2

    @staticmethod
    def extract_contact_point(
        mask_xy: np.ndarray,
        box_xyxy: Optional[np.ndarray] = None,
        y_band_percent: float = 2.0,
    ) -> Tuple[float, float]:
        mask_xy = np.asarray(mask_xy, dtype=float)

        if mask_xy.ndim != 2 or mask_xy.shape[0] < 3:
            return ContactPointPipeline._box_bottom_center(box_xyxy)

        ys = mask_xy[:, 1]
        height = float(ys.max() - ys.min())

        if height < 3.0:
            if box_xyxy is not None:
                return ContactPointPipeline._box_bottom_center(box_xyxy)
            return float(np.median(mask_xy[:, 0])), float(ys.max())

        band_thresh = ys.max() - max(1.0, height * (y_band_percent / 100.0))
        band = mask_xy[ys >= band_thresh]

        if len(band) == 0:
            if box_xyxy is not None:
                return ContactPointPipeline._box_bottom_center(box_xyxy)
            band = mask_xy[np.argmax(ys)][None, :]

        x = float(np.median(band[:, 0]))
        y = float(np.max(band[:, 1]))
        return x, y

    def validate_point(self, x: float, y: float) -> bool:
        if self.road_mask is None:
            return True
        xi, yi = int(round(x)), int(round(y))
        h, w = self.road_mask.shape[:2]
        if xi < 0 or yi < 0 or xi >= w or yi >= h:
            return False
        return bool(self.road_mask[yi, xi] > 0)

    def project_point(self, x: float, y: float) -> Tuple[Optional[float], Optional[float]]:
        if self.H is None:
            return None, None
        p = np.array([x, y, 1.0], dtype=float)
        q = self.H @ p
        if abs(q[2]) < 1e-9:
            return None, None
        return float(q[0] / q[2]), float(q[1] / q[2])

    def run(
        self,
        video_path: str,
        tracker: str = "bytetrack.yaml",
        conf: float = 0.25,
        max_frames: Optional[int] = None,
    ) -> pd.DataFrame:
        rows: List[TrackPoint] = []

        results = self.model.track(
            source=video_path,
            stream=True,
            persist=True,
            tracker=tracker,
            conf=conf,
            classes=self.classes,
            verbose=False,
        )

        for frame_idx, r in enumerate(results):
            if max_frames is not None and frame_idx >= max_frames:
                break

            if r.boxes is None or r.boxes.id is None:
                continue

            track_ids = r.boxes.id.int().cpu().tolist()
            cls_ids = r.boxes.cls.int().cpu().tolist() if r.boxes.cls is not None else [-1] * len(track_ids)
            confs = r.boxes.conf.cpu().tolist() if r.boxes.conf is not None else [0.0] * len(track_ids)
            boxes_xyxy = r.boxes.xyxy.cpu().numpy() if r.boxes.xyxy is not None else None
            masks_xy = r.masks.xy if r.masks is not None else [None] * len(track_ids)

            n = len(track_ids)
            for idx in range(n):
                mask_pts = masks_xy[idx] if idx < len(masks_xy) else None
                box_xyxy = boxes_xyxy[idx] if boxes_xyxy is not None and idx < len(boxes_xyxy) else None

                try:
                    if mask_pts is None:
                        x, y = self._box_bottom_center(box_xyxy)
                    else:
                        x, y = self.extract_contact_point(mask_pts, box_xyxy=box_xyxy)
                except Exception:
                    if box_xyxy is None:
                        continue
                    x, y = self._box_bottom_center(box_xyxy)

                is_valid = self.validate_point(x, y)
                wx, wy = self.project_point(x, y) if is_valid else (None, None)

                rows.append(
                    TrackPoint(
                        frame_idx=frame_idx,
                        track_id=int(track_ids[idx]),
                        cls_id=int(cls_ids[idx]),
                        conf=float(confs[idx]),
                        pixel_x=x,
                        pixel_y=y,
                        world_x=wx,
                        world_y=wy,
                        valid_road_mask=is_valid,
                    )
                )

        return pd.DataFrame([r.__dict__ for r in rows])

    def debug_annotate_video(
        self,
        video_path: str,
        output_path: str,
        tracker: str = "bytetrack.yaml",
        conf: float = 0.25,
        max_frames: int = 10,
    ) -> str:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        results = self.model.track(
            source=video_path,
            stream=True,
            persist=True,
            tracker=tracker,
            conf=conf,
            classes=self.classes,
            verbose=False,
        )

        for frame_idx, r in enumerate(results):
            if frame_idx >= max_frames:
                break

            frame = r.plot(labels=True, boxes=True, masks=True, probs=False)

            if r.boxes is not None and r.boxes.id is not None:
                boxes_xyxy = r.boxes.xyxy.cpu().numpy() if r.boxes.xyxy is not None else None
                masks_xy = r.masks.xy if r.masks is not None else []
                track_ids = r.boxes.id.int().cpu().tolist()

                for idx, tid in enumerate(track_ids):
                    mask_pts = masks_xy[idx] if idx < len(masks_xy) else None
                    box_xyxy = boxes_xyxy[idx] if boxes_xyxy is not None and idx < len(boxes_xyxy) else None

                    try:
                        if mask_pts is None:
                            x, y = self._box_bottom_center(box_xyxy)
                        else:
                            x, y = self.extract_contact_point(mask_pts, box_xyxy=box_xyxy)
                    except Exception:
                        continue

                    valid = self.validate_point(x, y)
                    color = (0, 255, 0) if valid else (0, 0, 255)

                    cv2.circle(frame, (int(round(x)), int(round(y))), 5, color, -1)
                    cv2.putText(
                        frame,
                        f"id={tid}",
                        (int(round(x)) + 6, int(round(y)) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                        cv2.LINE_AA,
                    )

            cv2.putText(
                frame,
                f"frame={frame_idx}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)

        writer.release()
        cap.release()
        return str(out_path)
