
import numpy as np
from typing import List, Tuple, Dict, Optional

def compute_centroid(box: np.ndarray) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])

def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return np.linalg.norm(p1 - p2)

def compute_pet(traj1: List[Tuple], traj2: List[Tuple],
                fps: float = 30.0,
                threshold_distance: float = 50.0) -> Optional[float]:
    """PET in seconds: time between first and second user of same spatial zone."""
    min_pet = None
    for t1 in traj1:
        _, f1, x1_1, y1_1, x2_1, y2_1, _, _ = t1
        c1 = compute_centroid(np.array([x1_1, y1_1, x2_1, y2_1]))
        for t2 in traj2:
            _, f2, x1_2, y1_2, x2_2, y2_2, _, _ = t2
            # enforce second vehicle AFTER first
            if f2 <= f1:
                continue
            c2 = compute_centroid(np.array([x1_2, y1_2, x2_2, y2_2]))
            dist = euclidean_distance(c1, c2)
            if dist < threshold_distance:
                pet_val = (f2 - f1) / fps  # seconds
                if min_pet is None or pet_val < min_pet:
                    min_pet = pet_val
    return min_pet

def compute_ttc(traj: List[Tuple], fps: float = 30.0) -> List[float]:
    if len(traj) < 2:
        return []
    ttc_values = []
    for i in range(len(traj) - 1):
        _, f1, x1_1, y1_1, x2_1, y2_1, _, _ = traj[i]
        _, f2, x1_2, y1_2, x2_2, y2_2, _, _ = traj[i+1]
        c1 = compute_centroid(np.array([x1_1, y1_1, x2_1, y2_1]))
        c2 = compute_centroid(np.array([x1_2, y1_2, x2_2, y2_2]))
        dist = euclidean_distance(c1, c2)
        time_diff = (f2 - f1) / fps
        if time_diff > 0:
            velocity = dist / time_diff
            if velocity > 0:
                ttc = dist / velocity
                ttc_values.append(ttc)
    return ttc_values

def analyze_conflicts(trajectories_dict: Dict[int, List],
                     fps: float = 30.0,
                     pet_threshold: float = 3.0) -> Dict:
    """PET threshold now in seconds."""
    track_ids = list(trajectories_dict.keys())
    conflicts = []

    for i, tid1 in enumerate(track_ids):
        for tid2 in track_ids[i+1:]:
            pet_val_sec = compute_pet(trajectories_dict[tid1],
                                      trajectories_dict[tid2],
                                      fps=fps)
            if pet_val_sec is not None and pet_val_sec < pet_threshold:
                conflicts.append({
                    "track_1": tid1,
                    "track_2": tid2,
                    "pet_seconds": pet_val_sec,
                    "severity": "high" if pet_val_sec < 1.5 else "medium"
                })

    ttc_stats = {}
    for tid, traj in trajectories_dict.items():
        ttc_vals = compute_ttc(traj, fps)
        if ttc_vals:
            ttc_stats[tid] = {
                "mean_ttc": float(np.mean(ttc_vals)),
                "min_ttc": float(np.min(ttc_vals)),
                "max_ttc": float(np.max(ttc_vals))
            }

    return {
        "conflicts": conflicts,
        "num_conflicts": len(conflicts),
        "ttc_statistics": ttc_stats,
        "tracks_analyzed": len(track_ids)
    }
