"""Convert pipeline output CSVs into the schemas the metric scripts expect.

Pure functions; no file I/O in this module.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.analysis.detection_metrics import Detection
from src.analysis.ssm_error import SsmEvent
from src.analysis.tracking_metrics import Track

# Pipeline columns the converter expects
_PIPELINE_DET_COLS: frozenset[str] = frozenset(
    {"frame", "track_id", "class_name", "conf", "x1", "y1", "x2", "y2", "cx", "cy"}
)
_PIPELINE_PET_COLS: frozenset[str] = frozenset({"orig_track_a", "orig_track_b", "pet"})


@dataclass(frozen=True)
class TrajPoint:
    frame: int
    track_id: int
    x: float
    y: float


def _check_columns(row: dict, required: frozenset[str]) -> None:
    missing = required - set(row.keys())
    if missing:
        raise ValueError(f"pipeline row missing columns: {sorted(missing)}")


def _to_float(x: object, default: float = 0.0) -> float:
    try:
        f = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if f != f:
        return default
    return f


def _to_int(x: object, default: int = -1) -> int:
    try:
        return int(float(x))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def detections_from_pipeline_rows(rows: Sequence[dict]) -> list[Detection]:
    """Convert pipeline detection rows to Detection objects."""
    dets: list[Detection] = []
    for r in rows:
        _check_columns(r, _PIPELINE_DET_COLS)
        dets.append(
            Detection(
                frame=_to_int(r["frame"]),
                box=(
                    _to_float(r["x1"]),
                    _to_float(r["y1"]),
                    _to_float(r["x2"]),
                    _to_float(r["y2"]),
                ),
                cls=str(r["class_name"]),
                conf=_to_float(r["conf"]),
            )
        )
    return dets


def tracks_from_pipeline_rows(rows: Sequence[dict]) -> list[Track]:
    """Convert pipeline detection rows to Track objects (pixel-space)."""
    tracks: list[Track] = []
    for r in rows:
        _check_columns(r, _PIPELINE_DET_COLS)
        x1 = _to_float(r["x1"])
        y1 = _to_float(r["y1"])
        x2 = _to_float(r["x2"])
        y2 = _to_float(r["y2"])
        tracks.append(
            Track(
                frame=_to_int(r["frame"]),
                track_id=_to_int(r["track_id"]),
                box=(x1, y1, x2, y2),
            )
        )
    return tracks


def _project_homography(H: np.ndarray, pt: tuple[float, float]) -> tuple[float, float]:
    v = H @ np.array([pt[0], pt[1], 1.0], dtype=np.float64)
    if abs(v[2]) < 1e-12:
        raise ValueError("homography produced a point at infinity")
    return float(v[0] / v[2]), float(v[1] / v[2])


def trajectories_from_pipeline_rows(rows: Sequence[dict], H: np.ndarray) -> list[TrajPoint]:
    """Project each detection's box center through H to world coordinates."""
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError("H must be 3x3")
    out: list[TrajPoint] = []
    for r in rows:
        _check_columns(r, _PIPELINE_DET_COLS)
        wx, wy = _project_homography(H, (_to_float(r["cx"]), _to_float(r["cy"])))
        out.append(
            TrajPoint(
                frame=_to_int(r["frame"]),
                track_id=_to_int(r["track_id"]),
                x=wx,
                y=wy,
            )
        )
    return out


def ssm_events_from_pet_rows(rows: Sequence[dict]) -> list[SsmEvent]:
    """Convert pipeline PET event rows to SsmEvent objects."""
    events: list[SsmEvent] = []
    for r in rows:
        _check_columns(r, _PIPELINE_PET_COLS)
        pet = r.get("pet")
        pet_f: float | None = None
        if pet is not None:
            f = _to_float(pet, default=float("nan"))
            pet_f = None if f != f else f
        ttc = r.get("ttc")
        ttc_f: float | None = None
        if ttc is not None:
            f = _to_float(ttc, default=float("nan"))
            ttc_f = None if f != f else f
        events.append(
            SsmEvent(
                track_a=_to_int(r["orig_track_a"]),
                track_b=_to_int(r["orig_track_b"]),
                pet=pet_f,
                ttc=ttc_f,
            )
        )
    return events


def traj_points_to_rows(points: Sequence[TrajPoint]) -> list[dict]:
    return [{"frame": p.frame, "track_id": p.track_id, "x": p.x, "y": p.y} for p in points]
