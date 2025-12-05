"""
Enhanced trajectory tracking with improved IoU matching and max_age=30
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict

def iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    """Calculate Intersection over Union (IoU) between two boxes"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / float(areaA + areaB - inter)

class TrajectoryTracker:
    """Enhanced tracker with IoU matching and max_age=30"""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks = []
        self.next_id = 0
        self.trajectories = {}
        self.trajectories_by_frame = defaultdict(list)

    def update(self, frame_idx: int, boxes: np.ndarray, 
               confidences: np.ndarray, class_ids: np.ndarray) -> Dict:
        matched_detections = set()
        matched_tracks = set()
        for track_id, track in enumerate(self.tracks):
            best_iou = 0
            best_detection_idx = -1
            for det_idx, box in enumerate(boxes):
                if det_idx in matched_detections:
                    continue
                current_iou = iou(track['box'], box)
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_detection_idx = det_idx
            if best_iou > self.iou_threshold:
                matched_detections.add(best_detection_idx)
                matched_tracks.add(track_id)
                self.tracks[track_id]['box'] = boxes[best_detection_idx]
                self.tracks[track_id]['confidence'] = confidences[best_detection_idx]
                self.tracks[track_id]['class_id'] = class_ids[best_detection_idx]
                self.tracks[track_id]['age'] = 0
                self.tracks[track_id]['frames_since_seen'] = 0
                tid = self.tracks[track_id]['id']
                if tid not in self.trajectories:
                    self.trajectories[tid] = []
                x1, y1, x2, y2 = boxes[best_detection_idx]
                self.trajectories[tid].append((
                    tid, frame_idx, x1, y1, x2, y2,
                    class_ids[best_detection_idx], confidences[best_detection_idx]
                ))
                self.trajectories_by_frame[frame_idx].append((
                    tid, x1, y1, x2, y2,
                    class_ids[best_detection_idx], confidences[best_detection_idx]
                ))
        for det_idx, box in enumerate(boxes):
            if det_idx not in matched_detections:
                new_track = {
                    'id': self.next_id,
                    'box': box,
                    'confidence': confidences[det_idx],
                    'class_id': class_ids[det_idx],
                    'age': 0,
                    'frames_since_seen': 0,
                    'start_frame': frame_idx
                }
                self.tracks.append(new_track)
                self.trajectories[self.next_id] = [(
                    self.next_id, frame_idx, box[0], box[1], box[2], box[3],
                    class_ids[det_idx], confidences[det_idx]
                )]
                self.trajectories_by_frame[frame_idx].append((
                    self.next_id, box[0], box[1], box[2], box[3],
                    class_ids[det_idx], confidences[det_idx]
                ))
                self.next_id += 1
        for track_id, track in enumerate(self.tracks):
            if track_id not in matched_tracks:
                self.tracks[track_id]['age'] += 1
                self.tracks[track_id]['frames_since_seen'] += 1
        self.tracks = [t for t in self.tracks if t['frames_since_seen'] <= self.max_age]
        return {
            'tracks': self.tracks,
            'trajectories': self.trajectories,
            'matched_count': len(matched_detections)
        }

    def get_trajectories(self) -> Dict:
        return self.trajectories

    def get_active_tracks(self) -> List[Dict]:
        return [t for t in self.tracks if t['frames_since_seen'] == 0]

    def get_track_count(self) -> int:
        return len(self.tracks)
