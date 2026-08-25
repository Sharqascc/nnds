"""
Constant acceleration baseline for trajectory prediction.
Uses last three points to estimate acceleration and predict future positions.
"""
import numpy as np


def constant_acceleration_predict(past_points, num_future=10):
    """
    past_points: list of (frame, x, y)
    Returns list of (frame, x, y)
    """
    if len(past_points) < 3:
        raise ValueError("Need at least 3 points")
    frames = [p[0] for p in past_points]
    xs = [p[1] for p in past_points]
    ys = [p[2] for p in past_points]

    # Estimate velocity and acceleration using finite differences
    dt = frames[-1] - frames[-2]
    vx = (xs[-1] - xs[-2]) / dt
    vy = (ys[-1] - ys[-2]) / dt
    ax = (vx - (xs[-2] - xs[-3]) / (frames[-2] - frames[-3])) / dt
    ay = (vy - (ys[-2] - ys[-3]) / (frames[-2] - frames[-3])) / dt

    future = []
    for i in range(1, num_future + 1):
        t = i
        future.append((
            frames[-1] + i,
            xs[-1] + vx * t + 0.5 * ax * t**2,
            ys[-1] + vy * t + 0.5 * ay * t**2
        ))
    return future


if __name__ == "__main__":
    past = [(0,0,0), (1,1,1), (2,4,4)]
    print(constant_acceleration_predict(past, 5))
