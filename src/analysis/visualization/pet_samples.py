"""Render N sample PET-pair images from a completed run.

Post-pipeline helper: reads a PET events CSV, detections CSV, and the
source video, then writes N annotated JPGs for the N lowest-PET events.

Each image contains:
  - Source video frame background
  - Grid overlay with the specific conflict cell highlighted (cyan)
  - Both tracks' trajectories as coloured polylines
  - Bounding boxes for both tracks at the conflict frame
  - Yellow ring at closest-approach point
  - Ring markers at the first actor's exit point (red) and second actor's
    entry point (blue) — the PET gap window
  - Info panel: sample number, CSV row, frame, PET (with arithmetic),
    conflict type, grid cell, track pair, entry/exit frames, frame gap

Called from ``traffic_analyzer.main()`` when ``--samples N`` is passed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

_COLOR_A = (0, 0, 255)  # track A — red
_COLOR_B = (255, 100, 0)  # track B — blue
_COLOR_MARK = (0, 255, 255)  # intersection — yellow
_COLOR_GRID_CELL = (255, 255, 0)  # highlighted cell — cyan
_COLOR_WHITE = (255, 255, 255)
_COLOR_BLACK = (0, 0, 0)


def _parse_traj(traj_json: Any) -> list[tuple[float, float, float]]:
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


def _try_load_grid(grid_config_path: str | Path | None):
    if not grid_config_path:
        return None
    try:
        from src.analysis.grid_trajectory.spatial_grid import SpatialGrid

        return SpatialGrid(str(grid_config_path))
    except Exception as e:
        print(f"[pet_samples] grid unavailable: {e}")
        return None


def _draw_grid_cell_overlay(frame: np.ndarray, grid, cell_id: str | None) -> np.ndarray:
    """Draw full grid + highlight the specific conflict cell (cyan)."""
    if grid is None:
        return frame
    try:
        frame = grid.draw_overlay(frame, alpha=0.3)
    except Exception:
        return frame
    if cell_id and cell_id != "unknown":
        try:
            center = grid.get_cell_center(cell_id)
            if center:
                cx, cy = int(center[0]), int(center[1])
                half = int(grid.cell_size) // 2
                overlay = frame.copy()
                cv2.rectangle(
                    overlay, (cx - half, cy - half), (cx + half, cy + half), _COLOR_GRID_CELL, -1
                )
                frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)
                cv2.rectangle(
                    frame, (cx - half, cy - half), (cx + half, cy + half), _COLOR_GRID_CELL, 2
                )
        except Exception:
            pass
    return frame


def _find_intersection_point(
    traj_a: list[tuple[float, float, float]],
    traj_b: list[tuple[float, float, float]],
) -> tuple[float, float] | None:
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


def _mark_frame_point(
    frame: np.ndarray,
    traj: list[tuple[float, float, float]],
    target_frame: int | None,
    color: tuple[int, int, int],
    label: str,
) -> bool:
    """Draw a ring at the trajectory point that occurs at ``target_frame``."""
    if target_frame is None:
        return False
    for p in traj:
        if int(p[2]) == int(target_frame):
            x, y = int(p[0]), int(p[1])
            cv2.circle(frame, (x, y), 12, color, 3)
            cv2.circle(frame, (x, y), 4, color, -1)
            w = frame.shape[1]
            lx = x + 16 if x < w - 240 else x - 220
            cv2.putText(frame, label, (lx, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            return True
    return False


def _draw_info_panel(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 2
    pad = 10
    line_h = 28
    text_w = max(
        (cv2.getTextSize(l, font, font_scale, thickness)[0][0] for l in lines), default=100
    )
    panel_w = text_w + 2 * pad
    panel_h = len(lines) * line_h + 2 * pad
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), _COLOR_BLACK, -1)
    frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
    for i, line in enumerate(lines):
        y = 10 + pad + (i + 1) * line_h - 8
        cv2.putText(frame, line, (10 + pad, y), font, font_scale, _COLOR_WHITE, thickness)
    return frame


def _fmt_opt(v, fmt: str, default: str = "?") -> str:
    return default if v is None else fmt.format(v)


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
    *,
    grid=None,
    event_idx: int = 0,
    total_events: int = 0,
    csv_row: int = 0,
    a_exit_frame: int | None = None,
    b_entry_frame: int | None = None,
    a_exit_time: float | None = None,
    b_entry_time: float | None = None,
) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None

    # 1. Grid overlay + cell highlight (bottom layer)
    frame = _draw_grid_cell_overlay(frame, grid, grid_cell)

    # Trajectories
    if len(traj_a) >= 2:
        _draw_polyline(frame, [(p[0], p[1]) for p in traj_a], _COLOR_A)
    if len(traj_b) >= 2:
        _draw_polyline(frame, [(p[0], p[1]) for p in traj_b], _COLOR_B)

    # Intersection marker (yellow ring)
    ip = _find_intersection_point(traj_a, traj_b)
    if ip:
        cx, cy = int(ip[0]), int(ip[1])
        cv2.circle(frame, (cx, cy), 18, _COLOR_MARK, 3)
        cv2.circle(frame, (cx, cy), 6, _COLOR_MARK, -1)

    # 3. Entry/exit frame markers (the PET gap window)
    _mark_frame_point(
        frame, traj_a, a_exit_frame, _COLOR_A, f"A out f={_fmt_opt(a_exit_frame, '{}', '?')}"
    )
    _mark_frame_point(
        frame, traj_b, b_entry_frame, _COLOR_B, f"B in f={_fmt_opt(b_entry_frame, '{}', '?')}"
    )

    # Bounding boxes
    if bbox_a:
        _draw_bbox(frame, bbox_a, _COLOR_A, f"T{track_a}")
    if bbox_b:
        _draw_bbox(frame, bbox_b, _COLOR_B, f"T{track_b}")

    # 2 + 4. Info panel with event ID, CSV row, PET arithmetic
    lines = [
        f"Sample {event_idx + 1}/{total_events}   CSV row {csv_row}",
        f"Frame: {frame_idx}    PET: {pet_val:.3f}s",
    ]
    if a_exit_time is not None and b_entry_time is not None:
        lines.append(f"  = B_in({b_entry_time:.2f}s) - A_out({a_exit_time:.2f}s)")
    else:
        lines.append("  = t_B_entry - t_A_exit")
    lines.append(f"Grid: {grid_cell or 'unknown'}    Type: {conflict_type or 'unknown'}")
    lines.append(f"Tracks: T{track_a} <-> T{track_b}")
    if a_exit_frame is not None or b_entry_frame is not None:
        lines.append(
            f"A out: f={_fmt_opt(a_exit_frame, '{}', '?')}    "
            f"B in: f={_fmt_opt(b_entry_frame, '{}', '?')}"
        )
    if a_exit_frame is not None and b_entry_frame is not None:
        gap = int(b_entry_frame) - int(a_exit_frame)
        lines.append(f"Gap: frames {a_exit_frame} -> {b_entry_frame}  ({gap}f)")

    frame = _draw_info_panel(frame, lines)
    return frame


def render_pet_samples(
    events_csv: str | Path,
    detections_csv: str | Path,
    video_path: str | Path,
    out_dir: str | Path,
    n: int = 5,
    grid_config_path: str | Path | None = None,
) -> list[Path]:
    """Render the N lowest-PET events as annotated JPG samples.

    Args:
        events_csv: path to the PET events CSV.
        detections_csv: path to the matching detections CSV.
        video_path: path to the source video.
        out_dir: directory to write images into.
        n: number of samples (default 5).
        grid_config_path: optional path to grid config JSON. If provided,
            the conflict cell is highlighted in the sample images.

    Returns:
        List of written paths, ordered by ascending PET value.
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
        .reset_index()  # preserves original CSV row in "index" column
    )

    det_lookup = _load_detections_lookup(detections_csv)
    grid = _try_load_grid(grid_config_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_events = len(df)
    written: list[Path] = []
    for rank, row in df.iterrows():
        frame_idx = int(row["_frame"])
        pet_val = float(row["_pet"])
        ta = int(row["_ta"])
        tb = int(row["_tb"])
        csv_row = int(row["index"])
        traj_a = _parse_traj(row.get("traj_a_json", "[]"))
        traj_b = _parse_traj(row.get("traj_b_json", "[]"))
        conflict_type = str(row.get("conflict_type", "unknown"))
        grid_cell = str(row.get("grid_cell", "unknown"))
        frame_dets = det_lookup.get(frame_idx, {})
        bbox_a = frame_dets.get(ta)
        bbox_b = frame_dets.get(tb)

        def _get_int(v):
            return None if pd.isna(v) else int(v)

        def _get_float(v):
            return None if pd.isna(v) else float(v)

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
            grid=grid,
            event_idx=rank,
            total_events=total_events,
            csv_row=csv_row,
            a_exit_frame=_get_int(row.get("track_a_exit_frame")),
            b_entry_frame=_get_int(row.get("track_b_entry_frame")),
            a_exit_time=_get_float(row.get("track_a_exit_time_sec")),
            b_entry_time=_get_float(row.get("track_b_entry_time_sec")),
        )
        if img is None:
            print(f"[pet_samples] skip event {rank}: frame {frame_idx} unreadable")
            continue

        fname = f"pet_sample_{rank:02d}_frame{frame_idx:04d}_pet{pet_val:.3f}_T{ta}xT{tb}.jpg"
        out_path = out_dir / fname
        cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        written.append(out_path)
        print(f"[pet_samples] wrote {out_path.name}")

    print(f"[pet_samples] {len(written)}/{total_events} samples -> {out_dir}")
    return written
