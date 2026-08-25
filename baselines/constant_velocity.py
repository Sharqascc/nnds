"""
Constant velocity baseline for trajectory prediction.

Given past trajectory points (x, y over time), predict future positions by
assuming constant velocity estimated from the last two points.
"""

import numpy as np


def constant_velocity_predict(past_points, num_future=10):
    """
    past_points: list of (frame, x, y)
    Returns: list of (frame, x, y) future points
    """
    if len(past_points) < 2:
        raise ValueError("Need at least 2 past points")
    last_frame, x_last, y_last = past_points[-1]
    prev_frame, x_prev, y_prev = past_points[-2]
    dt = last_frame - prev_frame
    if dt <= 0:
        raise ValueError("Frames must be strictly increasing")
    vx = (x_last - x_prev) / dt
    vy = (y_last - y_prev) / dt
    future = []
    for i in range(1, num_future + 1):
        future.append((last_frame + i, x_last + vx * i, y_last + vy * i))
    return future


if __name__ == "__main__":
    past = [(0, 0, 0), (1, 1, 2), (2, 2, 4)]
    future = constant_velocity_predict(past, 5)
    print(future)
