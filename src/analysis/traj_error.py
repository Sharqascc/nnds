"""Velocity and acceleration error metrics. Pure functions; no I/O.

Inputs are trajectories of (frame, x, y) in world coordinates (meters for NNDS)
plus fps. Velocity/acceleration are computed by central finite difference, so
errors are in m/s and m/s^2.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

Point = tuple[int, float, float]  # frame, x, y


def _require_fps(fps: float) -> None:
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a positive finite number")


def _sorted_by_frame(traj: Sequence[Point]) -> list[Point]:
    return sorted(traj, key=lambda p: p[0])


def velocity(traj: Sequence[Point], fps: float) -> list[tuple[int, float, float]]:
    """Central-difference velocity (vx, vy) in m/s at interior frames."""
    _require_fps(fps)
    pts = _sorted_by_frame(traj)
    if len(pts) < 2:
        return []
    out: list[tuple[int, float, float]] = []
    for i in range(1, len(pts) - 1):
        f_prev, x_prev, y_prev = pts[i - 1]
        f_next, x_next, y_next = pts[i + 1]
        dt = (f_next - f_prev) / fps
        if dt <= 0:
            continue
        out.append(
            (
                pts[i][0],
                (x_next - x_prev) / dt,
                (y_next - y_prev) / dt,
            )
        )
    # forward/backward for endpoints
    dt0 = (pts[1][0] - pts[0][0]) / fps
    if dt0 > 0:
        out.insert(0, (pts[0][0], (pts[1][1] - pts[0][1]) / dt0, (pts[1][2] - pts[0][2]) / dt0))
    dtn = (pts[-1][0] - pts[-2][0]) / fps
    if dtn > 0:
        out.append((pts[-1][0], (pts[-1][1] - pts[-2][1]) / dtn, (pts[-1][2] - pts[-2][2]) / dtn))
    return out


def speed(traj: Sequence[Point], fps: float) -> list[tuple[int, float]]:
    """Scalar speed (m/s) per frame."""
    return [(f, float(np.hypot(vx, vy))) for f, vx, vy in velocity(traj, fps)]


def acceleration(traj: Sequence[Point], fps: float) -> list[tuple[int, float, float]]:
    """Central-difference acceleration (ax, ay) in m/s^2 at interior frames."""
    _require_fps(fps)
    pts = _sorted_by_frame(traj)
    if len(pts) < 3:
        return []
    out: list[tuple[int, float, float]] = []
    for i in range(1, len(pts) - 1):
        f_prev, x_prev, y_prev = pts[i - 1]
        f_cur, x_cur, y_cur = pts[i]
        f_next, x_next, y_next = pts[i + 1]
        dt = (f_next - f_prev) / fps
        if dt <= 0:
            continue
        ax = (x_next - 2 * x_cur + x_prev) / ((dt / 2.0) ** 2)
        ay = (y_next - 2 * y_cur + y_prev) / ((dt / 2.0) ** 2)
        out.append((f_cur, ax, ay))
    return out


def _pair_on_common_frames(
    pred: Sequence[tuple[int, float]],
    gt: Sequence[tuple[int, float]],
) -> tuple[np.ndarray, np.ndarray]:
    gd = {f: v for f, v in gt}
    common = [(f, pv, gd[f]) for f, pv in pred if f in gd]
    if not common:
        return np.zeros((0,)), np.zeros((0,))
    p = np.array([c[1] for c in common], dtype=np.float64)
    g = np.array([c[2] for c in common], dtype=np.float64)
    return p, g


def velocity_metrics(
    pred_traj: Sequence[Point], gt_traj: Sequence[Point], fps: float
) -> dict[str, float]:
    """MAE and RMSE of scalar speed (m/s) on common frames."""
    p, g = _pair_on_common_frames(speed(pred_traj, fps), speed(gt_traj, fps))
    if p.size == 0:
        return {"mae": 0.0, "rmse": 0.0, "n": 0}
    errs = np.abs(p - g)
    return {
        "mae": float(np.mean(errs)),
        "rmse": float(np.sqrt(np.mean(errs**2))),
        "n": int(errs.size),
    }


def acceleration_metrics(
    pred_traj: Sequence[Point], gt_traj: Sequence[Point], fps: float
) -> dict[str, float]:
    """MAE and RMSE of scalar acceleration magnitude (m/s^2) on common frames."""
    pred = [(f, float(np.hypot(ax, ay))) for f, ax, ay in acceleration(pred_traj, fps)]
    gt = [(f, float(np.hypot(ax, ay))) for f, ax, ay in acceleration(gt_traj, fps)]
    p, g = _pair_on_common_frames(pred, gt)
    if p.size == 0:
        return {"mae": 0.0, "rmse": 0.0, "n": 0}
    errs = np.abs(p - g)
    return {
        "mae": float(np.mean(errs)),
        "rmse": float(np.sqrt(np.mean(errs**2))),
        "n": int(errs.size),
    }


def trajectory_metrics(
    pred_traj: Sequence[Point], gt_traj: Sequence[Point], fps: float
) -> dict[str, dict[str, float]]:
    """Combined velocity + acceleration metrics."""
    return {
        "velocity": velocity_metrics(pred_traj, gt_traj, fps),
        "acceleration": acceleration_metrics(pred_traj, gt_traj, fps),
    }
