"""
Deterministic geometric conflict type classifier.

Classifies PET events into:
  - rear_end
  - head_on
  - crossing
  - side_swipe
  - other

based on velocity vector angles and relative motion derived from the trajectory JSON.

Trajectory JSON format: list of dicts with keys 'frame', 'x_pixel', 'y_pixel',
'world_x', 'world_y' (world coordinates optional).
"""

import math
from typing import List, Dict, Tuple


def _get_velocity_vector(points: List[Dict], before_frame: int, window: int = 5) -> Tuple[float, float]:
    """Compute average velocity vector from the last `window` points before `before_frame`."""
    # Filter points before before_frame, sorted by frame
    pts = [p for p in points if p.get('frame', 0) < before_frame]
    if len(pts) < 2:
        return (0.0, 0.0)
    pts = sorted(pts, key=lambda p: p['frame'])[-window:]
    if len(pts) < 2:
        return (0.0, 0.0)

    # Use last two points to estimate velocity (simplest robust approach)
    p1 = pts[0]
    p2 = pts[-1]
    dt = p2['frame'] - p1['frame']
    if dt <= 0:
        return (0.0, 0.0)

    # Prefer world coordinates if available, else pixel
    if all(p1.get(k) is not None and p2.get(k) is not None for k in ['world_x', 'world_y']):
        vx = (p2['world_x'] - p1['world_x']) / dt
        vy = (p2['world_y'] - p1['world_y']) / dt
    else:
        vx = (p2['x_pixel'] - p1['x_pixel']) / dt
        vy = (p2['y_pixel'] - p1['y_pixel']) / dt

    return (vx, vy)


def _angle_between(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    """Angle between two 2D vectors in degrees [0, 180]."""
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.hypot(*v1)
    mag2 = math.hypot(*v2)
    if mag1 == 0 or mag2 == 0:
        return 0.0
    cos_theta = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_theta))


def classify_conflict_geometry(traj_a_json: str, traj_b_json: str,
                               conflict_frame: int, fps: float = 30.0) -> str:
    """
    Classify conflict type from trajectory JSONs and conflict frame.

    Returns one of: 'rear_end', 'head_on', 'crossing', 'side_swipe', 'other'
    """
    import json

    if not traj_a_json or not traj_b_json:
        return 'other'

    try:
        pts_a = json.loads(traj_a_json)
        pts_b = json.loads(traj_b_json)
    except Exception:
        return 'other'

    v_a = _get_velocity_vector(pts_a, conflict_frame, window=5)
    v_b = _get_velocity_vector(pts_b, conflict_frame, window=5)

    if v_a == (0, 0) or v_b == (0, 0):
        return 'other'

    angle = _angle_between(v_a, v_b)

    # Relative speed magnitude (for side-swipe criterion)
    speed_a = math.hypot(*v_a)
    speed_b = math.hypot(*v_b)
    speed_diff = abs(speed_a - speed_b)

    # Same direction if dot > 0 and angle small
    if angle < 30.0:
        # Could be rear-end or side-swipe depending on lateral offset
        # For simplicity, we'll classify as rear_end if speed difference is significant
        # and side_swipe if speeds similar (parallel)
        if speed_diff > 0.5 * max(speed_a, speed_b):
            return 'rear_end'
        else:
            return 'side_swipe'
    elif angle > 150.0:
        return 'head_on'
    elif 60.0 <= angle <= 120.0:
        return 'crossing'
    else:
        return 'other'
