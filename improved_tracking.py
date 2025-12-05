
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict

def iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
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
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 5):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks = []
        self.next_id = 0
        self.trajectories = []
        self.trajectories_by_frame = defaultdict(list)
        
    def update(self, detections: Dict, frame_idx: int) -> List[Tuple]:
        boxes = detections["boxes"]
        confs = detections["confs"]
        classes = detections["classes"]
        
        for tr in self.tracks:
            tr["age"] += 1
        
        used_track_ids = set()
        frame_tracks = []
        
        for box, conf, cls in zip(boxes, confs, classes):
            best_iou, best_tr = 0.0, None
            for tr in self.tracks:
                if tr["age"] > self.max_age:
                    continue
                if tr["last_frame"] != frame_idx - 1:
                    continue
                i = iou(box, tr["last_box"])
                if i > best_iou:
                    best_iou, best_tr = i, tr
            
            if best_iou > self.iou_threshold and best_tr is not None and best_tr["id"] not in used_track_ids:
                tid = best_tr["id"]
                best_tr["last_box"] = box
                best_tr["last_frame"] = frame_idx
                best_tr["age"] = 0
                used_track_ids.add(tid)
            else:
                tid = self.next_id
                self.tracks.append({
                    "id": tid,
                    "last_box": box,
                    "last_frame": frame_idx,
                    "age": 0,
                    "class": int(cls)
                })
                used_track_ids.add(tid)
                self.next_id += 1
            
            traj_point = (tid, frame_idx, *box, int(cls), float(conf))
            self.trajectories.append(traj_point)
            self.trajectories_by_frame[frame_idx].append(traj_point)
            frame_tracks.append(traj_point)
        
        self.tracks = [tr for tr in self.tracks if tr["age"] <= self.max_age]
        return frame_tracks
    
    def get_track_trajectories(self) -> Dict[int, List]:
        traj_dict = defaultdict(list)
        for traj_point in self.trajectories:
            tid = traj_point[0]
            traj_dict[tid].append(traj_point)
        return dict(traj_dict)
    
    def get_active_tracks(self) -> List[int]:
        return [tr["id"] for tr in self.tracks if tr["age"] == 0]
    
    def get_statistics(self) -> Dict:
        traj_dict = self.get_track_trajectories()
        return {
            "total_tracks": len(traj_dict),
            "active_tracks": len([tr for tr in self.tracks if tr["age"] == 0]),
            "trajectory_points": sum(len(v) for v in traj_dict.values()),
            "frames_processed": len(self.trajectories_by_frame)
        }
