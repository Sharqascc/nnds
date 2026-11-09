"""Render N sample PET-pair images from a completed run.

Post-pipeline helper: reads a PET events CSV, the matching detections CSV,
and the source video, then writes N annotated JPGs for the N lowest-PET
events. Each image contains:

  - Frame background from the source video
  - Both tracks' trajectories as thin coloured polylines
  - Bounding boxes for both tracks at the conflict frame
  - A yellow ring marking the closest approach (trajectory intersection)
  - Info panel: frame, PET, conflict type, grid cell, track pair

Called from ``traffic_analyzer.main()`` when ``--samples N`` is passed.
Does not require ``VideoOverlayPlotter`` — everything is drawn with ``cv2``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

_COLOR_A = (0, 0, 255)  # track A — red (BGR)
_COLOR_B = (255, 100, 0)  # track B — blue
_COLOR_MARK = (0, 255, 255)  # intersection — yellow
_COLOR_WHITE = (255, 255, 255)
_COLOR_BLACK = (0, 0, 0)


def _parse_traj(traj_json: Any) -> list[tuple[float, float, float]]:
    """Parse traj JSON to [(x_pixel, y_pixel, frame), ...]."""
    try:
        pts = json.loads(traj_json) if isinstance(traj_json, str) else traj_json
    except (json.JSONDecodeError, TypeError):
        return []
    out: list[tuple[float, float, float]] = []
    for p in pts or []:
        try:
            out.append((float(p["x_pixel"]), float(p["y_pixel"]), float(p["frame"])))
        except (KeyError, ValueError, TypeError):
            continue
    return out


def _load_detections_lookup(
    csv_path: Path,
) -> dict[int, dict[int, tuple[int, int, int, int]]]:
    """Return {frame: {track_id: (x1, y1, x2, y2)}}."""
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    out: dict[int, dict[int, tuple[int, int, int, int]]] = {}
    for _, row in df.iterrows():
        try:
            frame = int(row["frame"])
            tid = int(row["track_id"])
            x1 = int(row["x1"])
            y1 = int(row["y1"])
            x2 = int(row["x2"])
            y2 = int(row["y2"])
        except (KeyError, ValueError, TypeError):
            continue
        out.setdefault(frame, {})[tid] = (x1, y1, x2, y2)
    return out


def _find_intersection_point(
    traj_a: list[tuple[float, float, float]],
    traj_b: list[tuple[float, float, float]],
) -> tuple[float, float] | None:
    """Return (x, y) mid-point of closest approach between two trajectories."""
    if not traj_a or not traj_b:
        return None
    best_xy: tuple[float, float] | None = None
    best_d2 = float("inf")
    for ax, ay, _ in traj_a:
        for bx, by, _ in traj_b:
            d2 = (ax - bx) ** 2 + (ay - by) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_xy = ((ax + bx) / 2.0, (ay + by) / 2.0)
    return best_xy


def _draw_polyline(
    frame: np.ndarray,
    points: list[tuple[float, float]],
    color: tuple[int, int, int],
) -> None:
    pts = [(int(p[0]), int(p[1])) for p in points]
    for i in range(len(pts) - 1):
        cv2.line(frame, pts[i], pts[i + 1], color, 2)


def _draw_bbox(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: tuple[int, int, int],
    label: str,
) -> None:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(label, font, 0.7, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 6), font, 0.7, _COLOR_WHITE, 2)


def _draw_info_panel(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Draw semi-transparent black panel top-left with multiple text lines."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    pad = 10
    line_h = 30
    text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in lines]
    panel_w = max(text_sizes) + 2 * pad if text_sizes else 100
    panel_h = len(lines) * line_h + 2 * pad
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), _COLOR_BLACK, -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    for i, line in enumerate(lines):
        y = 10 + pad + (i + 1) * line_h - 8
        cv2.putText(frame, line, (10 + pad, y), font, font_scale, _COLOR_WHITE, thickness)
    return frame


def _render_one_sample(
    video_path: Path,
    frame_idx: int,
    traj_a: list[tuple[float, float, float]],
    traj_b: list[tuple[float, float, float]],
    track_a: int,
    track_b: int,
    bbox_a: tuple[int, int, int, int] | None,
    bbox_b: tuple[int, int, int, int] | None,
    pet_val: float,
    conflict_type: str,
    grid_cell: str,
) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None

    if len(traj_a) >= 2:
        _draw_polyline(frame, [(p[0], p[1]) for p in traj_a], _COLOR_A)
    if len(traj_b) >= 2:
        _draw_polyline(frame, [(p[0], p[1]) for p in traj_b], _COLOR_B)

    ip = _find_intersection_point(traj_a, traj_b)
    if ip:
        cx, cy = int(ip[0]), int(ip[1])
        cv2.circle(frame, (cx, cy), 18, _COLOR_MARK, 3)
        cv2.circle(frame, (cx, cy), 6, _COLOR_MARK, -1)

    if bbox_a:
        _draw_bbox(frame, bbox_a, _COLOR_A, f"T{track_a}")
    if bbox_b:
        _draw_bbox(frame, bbox_b, _COLOR_B, f"T{track_b}")

    lines = [
        f"Frame:   {frame_idx}",
        f"PET:     {pet_val:.3f}s",
        f"Type:    {conflict_type or 'unknown'}",
        f"Grid:    {grid_cell or 'unknown'}",
        f"Tracks:  T{track_a} <-> T{track_b}",
    ]
    frame = _draw_info_panel(frame, lines)
    return frame


def render_pet_samples(
    events_csv: str | Path,
    detections_csv: str | Path,
    video_path: str | Path,
    out_dir: str | Path,
    n: int = 5,
) -> list[Path]:
    """Render the N lowest-PET events as annotated JPG samples.

    Args:
        events_csv: path to the PET events CSV from the run.
        detections_csv: path to the matching detections CSV.
        video_path: path to the source video.
        out_dir: directory to write sample images into (created if absent).
        n: number of samples to write (default 5).

    Returns:
        List of written paths, ordered by ascending PET value.

    Raises:
        FileNotFoundError: if events_csv or video_path does not exist.
        ValueError: if events_csv is missing required columns.
    """
    events_csv = Path(events_csv)
    detections_csv = Path(detections_csv)
    video_path = Path(video_path)
    out_dir = Path(out_dir)

    if not events_csv.exists():
        raise FileNotFoundError(f"events CSV not found: {events_csv}")
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    df = pd.read_csv(events_csv)
    for col in ("pet", "frame", "track_a", "track_b"):
        if col not in df.columns:
            raise ValueError(f"events CSV missing required column: {col}")
    if df.empty:
        print("[pet_samples] no PET events - nothing to render")
        return []

    df = (
        df.assign(
            _frame=df["frame"].astype(int),
            _ta=df["track_a"].astype(int),
            _tb=df["track_b"].astype(int),
            _pet=df["pet"].astype(float),
        )
        .sort_values("_pet", ascending=True)
        .head(n)
        .reset_index(drop=True)
    )

    det_lookup = _load_detections_lookup(detections_csv)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for idx, row in df.iterrows():
        frame_idx = int(row["_frame"])
        pet_val = float(row["_pet"])
        ta = int(row["_ta"])
        tb = int(row["_tb"])
        traj_a = _parse_traj(row.get("traj_a_json", "[]"))
        traj_b = _parse_traj(row.get("traj_b_json", "[]"))
        conflict_type = str(row.get("conflict_type", "unknown"))
        grid_cell = str(row.get("grid_cell", "unknown"))
        frame_dets = det_lookup.get(frame_idx, {})
        bbox_a = frame_dets.get(ta)
        bbox_b = frame_dets.get(tb)

        img = _render_one_sample(
            video_path,
            frame_idx,
            traj_a,
            traj_b,
            ta,
            tb,
            bbox_a,
            bbox_b,
            pet_val,
            conflict_type,
            grid_cell,
        )
        if img is None:
            print(f"[pet_samples] skip event {idx}: frame {frame_idx} unreadable")
            continue

        fname = f"pet_sample_{idx:02d}_frame{frame_idx:04d}_pet{pet_val:.3f}_T{ta}xT{tb}.jpg"
        out_path = out_dir / fname
        cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        written.append(out_path)
        print(f"[pet_samples] wrote {out_path.name}")

    print(f"[pet_samples] {len(written)}/{len(df)} samples -> {out_dir}")
    return written
