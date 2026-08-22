"""
Custom motion-based multi-object tracker.
Uses Kalman filters for motion prediction and Hungarian matching.
Two-stage association: IoU + predicted-center recovery.
"""

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass


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


class KalmanTrack:
    """Kalman filter state for a single object."""

    def __init__(self, det: Detection, track_id: int):
        self.id = track_id
        self.cls_id = det.cls_id
        self.cls_name = det.cls_name
        self.source = det.source
        self.age = 0
        self.time_since_update = 0

        w = max(det.x2 - det.x1, 1.0)
        h = max(det.y2 - det.y1, 1.0)
        self.kf = cv2.KalmanFilter(6, 4)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
        ], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.1
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 10.0
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)
        self.kf.statePost = np.array([det.cx, det.cy, w, h, 0, 0], dtype=np.float32).reshape(-1, 1)

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, det: Detection):
        measurement = np.array([det.cx, det.cy, max(det.x2 - det.x1, 1), max(det.y2 - det.y1, 1)], dtype=np.float32).reshape(-1, 1)
        self.kf.correct(measurement)
        self.time_since_update = 0

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
    """Kalman + Hungarian tracker for stable IDs."""

    def __init__(self, max_age=60, min_hits=1, iou_threshold=0.2):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
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

    def update(self, detections):
        """Update tracks with new detections; returns dict det_index -> track_id."""
        for t in self.tracks.values():
            t.predict()

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

        for i, j in zip(row_ind, col_ind):
            if cost[i, j] < 1.0 - self.iou_threshold:  # IoU > threshold
                tid = active_ids[i]
                self.tracks[tid].update(detections[j])
                matched[j] = tid
                matched_track_ids.add(tid)
                unmatched_dets.discard(j)

        # Stage 2: predicted-center distance matching for remaining detections/tracks
        remaining_dets = [j for j in unmatched_dets]
        remaining_tracks = [tid for tid in self.tracks if tid not in matched_track_ids]

        if remaining_dets and remaining_tracks:
            cost2 = np.zeros((len(remaining_tracks), len(remaining_dets)), dtype=np.float32)
            for i, tid in enumerate(remaining_tracks):
                pred_center = self.tracks[tid].center
                for j, det_idx in enumerate(remaining_dets):
                    det = detections[det_idx]
                    dist = np.sqrt((pred_center[0]-det.cx)**2 + (pred_center[1]-det.cy)**2)
                    cost2[i, j] = dist

            row_ind2, col_ind2 = linear_sum_assignment(cost2)
            for i, j in zip(row_ind2, col_ind2):
                if cost2[i, j] < 80.0:
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
