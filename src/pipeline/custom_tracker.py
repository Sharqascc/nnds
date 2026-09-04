"""
Custom motion-based multi-object tracker with appearance disambiguation.

Uses:
  - Kalman filters for motion prediction
  - Hungarian matching (IoU first, then predicted-center + appearance)
  - HSV histograms and deep ReID embeddings for occlusion handling
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Detection:
    frame: int
    x1: float
    y1: float
    x2: float
    y2: float
    cx: float
    cy: float
    cls_id: int
    cls_name: str
    conf: float
    source: str
    hist: np.ndarray | None = None
    embedding: np.ndarray | None = None


class KalmanTrack:
    """Kalman filter state for a single object."""

    def __init__(self, det: Detection, track_id: int):
        self.id = track_id
        self.cls_id = det.cls_id
        self.cls_name = det.cls_name
        self.source = det.source
        self.age = 0
        self.time_since_update = 0
        self.hist = det.hist if det.hist is not None else None
        self.embedding = det.embedding if det.embedding is not None else None

        w = max(det.x2 - det.x1, 1.0)
        h = max(det.y2 - det.y1, 1.0)
        self.kf = cv2.KalmanFilter(6, 4)
        self.kf.transitionMatrix = np.array(
            [
                [1, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 1],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        self.kf.measurementMatrix = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
            ],
            dtype=np.float32,
        )
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.1
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 10.0
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)
        self.kf.statePost = np.array([det.cx, det.cy, w, h, 0, 0], dtype=np.float32).reshape(-1, 1)

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, det: Detection):
        measurement = np.array(
            [det.cx, det.cy, max(det.x2 - det.x1, 1), max(det.y2 - det.y1, 1)],
            dtype=np.float32,
        ).reshape(-1, 1)
        self.kf.correct(measurement)
        self.time_since_update = 0
        if det.hist is not None:
            if self.hist is None:
                self.hist = det.hist
            else:
                self.hist = 0.8 * self.hist + 0.2 * det.hist
        if det.embedding is not None:
            if self.embedding is None:
                self.embedding = det.embedding
            else:
                self.embedding = 0.8 * self.embedding + 0.2 * det.embedding

    @property
    def center(self):
        s = np.asarray(self.kf.statePost).reshape(-1)
        return s[0], s[1]

    @property
    def box(self):
        s = np.asarray(self.kf.statePost).reshape(-1)
        w = max(s[2], 1)
        h = max(s[3], 1)
        return (s[0] - w / 2, s[1] - h / 2, s[0] + w / 2, s[1] + h / 2)


class CustomTracker:
    """Kalman + Hungarian tracker with appearance matching."""

    def __init__(
        self,
        max_age=60,
        min_hits=1,
        iou_threshold=0.2,
        log_overlaps=False,
        overlap_log_path="outputs/tracking_overlap_debug.log",
        reid_encoder=None,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.log_overlaps = log_overlaps
        self.reid_encoder = reid_encoder
        if log_overlaps and overlap_log_path:
            Path(overlap_log_path).parent.mkdir(parents=True, exist_ok=True)
            self.log_handle = open(overlap_log_path, "w", encoding="utf-8")
            self.log_handle.write(
                "frame,track_a,track_b,iou,pred_cx_a,pred_cy_a,pred_cx_b,pred_cy_b,box_a,box_b\n"
            )
        else:
            self.log_handle = None
        self.next_id = 1
        self.tracks = {}

    def _iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
        area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def _cosine_similarity(self, a, b):
        if a is None or b is None:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))

    def _appearance_cost(self, track_hist, det_hist, track_emb, det_emb):
        """Return a cost (0=perfect similarity) between track and detection appearance."""
        # Use deep embedding if both available
        if track_emb is not None and det_emb is not None:
            sim = self._cosine_similarity(track_emb, det_emb)
            return (1.0 - sim) * 100.0  # scale cost
        # Fallback to HSV histogram if available
        if track_hist is not None and det_hist is not None:
            return (
                cv2.compareHist(
                    track_hist.astype(np.float32),
                    det_hist.astype(np.float32),
                    cv2.HISTCMP_BHATTACHARYYA,
                )
                * 50.0
            )
        return 0.0

    def update(self, detections, frame_img=None, frame=None):
        """Update tracks with new detections; returns dict det_index -> track_id."""
        for t in self.tracks.values():
            t.predict()

        if self.log_overlaps and self.log_handle is not None:
            frame_now = frame if frame is not None else (detections[0].frame if detections else -1)
            track_ids = list(self.tracks.keys())
            pred_boxes = {tid: self.tracks[tid].box for tid in track_ids}
            centers = {tid: self.tracks[tid].center for tid in track_ids}
            for i in range(len(track_ids)):
                for j in range(i + 1, len(track_ids)):
                    tid_a = track_ids[i]
                    tid_b = track_ids[j]
                    iou = self._iou(pred_boxes[tid_a], pred_boxes[tid_b])
                    if iou > 0.5:
                        cx_a, cy_a = centers[tid_a]
                        cx_b, cy_b = centers[tid_b]
                        box_a = ",".join(f"{v:.1f}" for v in pred_boxes[tid_a])
                        box_b = ",".join(f"{v:.1f}" for v in pred_boxes[tid_b])
                        self.log_handle.write(
                            f"{frame_now},{tid_a},{tid_b},{iou:.3f},{cx_a:.1f},{cy_a:.1f},{cx_b:.1f},{cy_b:.1f},{box_a},{box_b}\n"
                        )

        if not detections:
            return {}

        active_ids = list(self.tracks.keys())

        # Stage 1: IoU matching with predicted boxes
        cost = np.zeros((len(active_ids), len(detections)), dtype=np.float32)
        for i, tid in enumerate(active_ids):
            pred_box = self.tracks[tid].box
            for j, det in enumerate(detections):
                det_box = (det.x1, det.y1, det.x2, det.y2)
                iou = self._iou(pred_box, det_box)
                cost[i, j] = 1.0 - iou

        row_ind, col_ind = linear_sum_assignment(cost)
        matched = {}
        matched_track_ids = set()
        unmatched_dets = set(range(len(detections)))

        for i, j in zip(row_ind, col_ind, strict=False):
            if cost[i, j] < 1.0 - self.iou_threshold:  # IoU > threshold
                tid = active_ids[i]
                self.tracks[tid].update(detections[j])
                matched[j] = tid
                matched_track_ids.add(tid)
                unmatched_dets.discard(j)

        # Stage 2: predicted-center + appearance matching for remaining detections/tracks
        remaining_dets = [j for j in unmatched_dets]
        remaining_tracks = [tid for tid in self.tracks if tid not in matched_track_ids]

        if remaining_dets and remaining_tracks:
            # Compute embeddings for remaining detections on the fly if encoder available
            det_embeddings = {}
            for j in remaining_dets:
                det = detections[j]
                if (
                    self.reid_encoder is not None
                    and frame_img is not None
                    and det.embedding is None
                ):
                    det.embedding = self.reid_encoder.encode_crop(
                        frame_img, det.x1, det.y1, det.x2, det.y2
                    )
                det_embeddings[j] = det.embedding

            cost2 = np.zeros((len(remaining_tracks), len(remaining_dets)), dtype=np.float32)
            for i, tid in enumerate(remaining_tracks):
                pred_center = self.tracks[tid].center
                track_hist = self.tracks[tid].hist
                track_emb = self.tracks[tid].embedding
                for j, det_idx in enumerate(remaining_dets):
                    det = detections[det_idx]
                    dist = np.sqrt((pred_center[0] - det.cx) ** 2 + (pred_center[1] - det.cy) ** 2)
                    app_cost = self._appearance_cost(
                        track_hist, det.hist, track_emb, det_embeddings[det_idx]
                    )
                    cost2[i, j] = dist + app_cost

            row_ind2, col_ind2 = linear_sum_assignment(cost2)
            for i, j in zip(row_ind2, col_ind2, strict=False):
                if cost2[i, j] < 150.0:  # threshold increased due to appearance cost scale
                    tid = remaining_tracks[i]
                    det_idx = remaining_dets[j]
                    self.tracks[tid].update(detections[det_idx])
                    matched[det_idx] = tid
                    unmatched_dets.discard(det_idx)

        # Create new tracks for still unmatched detections
        for j in unmatched_dets:
            det = detections[j]
            new_track = KalmanTrack(det, self.next_id)
            self.tracks[self.next_id] = new_track
            matched[j] = self.next_id
            self.next_id += 1

        # Remove dead tracks
        self.tracks = {
            tid: t for tid, t in self.tracks.items() if t.time_since_update < self.max_age
        }

        return matched
