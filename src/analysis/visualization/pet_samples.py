"""Render N sample PET-pair images from a completed run.

Post-pipeline helper: reads a PET events CSV, detections CSV, and the
source video, then writes N annotated JPGs for the N lowest-PET events.

Publication-quality styling:
  - Localized grid drawn ONLY within N cells of the conflict cell, at
    15% opacity, gray (not neon), coarser spacing (every 2 cells).
  - Cell highlight drawn AFTER the grid as a solid cyan border with a
    light fill — never hidden under the mesh.
  - Track-ID labels use collision resolution: when boxes overlap, labels
    are offset vertically so none overlap. All labels have a dark badge.
  - Trajectories carry an arrowhead showing direction of travel.
  - Info panel is a rounded semi-transparent card with title / PET
    arithmetic / structured metadata — not a bare text dump on the road.

Called from ``traffic_analyzer.main()`` when ``--samples N`` is passed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

# ── Palette ──────────────────────────────────────────────────────────────
_COLOR_A = (0, 0, 255)  # track A — red
_COLOR_B = (255, 100, 0)  # track B — blue
_COLOR_MARK = (0, 255, 255)  # intersection ring — yellow
_COLOR_GRID_CELL = (255, 255, 0)  # conflict cell — cyan
_COLOR_GRID_LINE = (140, 140, 140)  # subtle gray
_COLOR_WHITE = (255, 255, 255)
_COLOR_BLACK = (0, 0, 0)
_COLOR_PANEL_BG = (30, 30, 30)
_COLOR_PANEL_BORDER = (100, 100, 100)
_COLOR_PANEL_TEXT = (225, 225, 225)
_COLOR_PANEL_TITLE = (255, 255, 255)

# ── Style constants ──────────────────────────────────────────────────────
_GRID_ALPHA = 0.15
_GRID_LINE_THICKNESS = 1
_GRID_SKIP = 2  # draw every Nth cell boundary
_LOCAL_RADIUS_CELLS = 4  # grid drawn within +/- N cells of conflict
_LABEL_SCALE = 0.55
_LABEL_THICKNESS = 2


# ══════════════════════════════════════════════════════════════════════════
# Parsers
# ══════════════════════════════════════════════════════════════════════════


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
            x1, y1 = int(row["x1"]), int(row["y1"])
            x2, y2 = int(row["x2"]), int(row["y2"])
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


# ══════════════════════════════════════════════════════════════════════════
# Drawing helpers
# ══════════════════════════════════════════════════════════════════════════


def _draw_localized_grid(
    frame: np.ndarray,
    grid,
    conflict_cell: str | None,
    *,
    radius_cells: int = _LOCAL_RADIUS_CELLS,
    alpha: float = _GRID_ALPHA,
    skip: int = _GRID_SKIP,
) -> np.ndarray:
    """Draw grid lines only within a small radius around the conflict cell.

    Much less visually intrusive than a full-frame mesh — the reviewer's
    "moire" complaint comes from a 25-pixel mesh across the whole frame
    at bright yellow. This version draws ~9x9 cells at 15% alpha in gray.
    """
    if grid is None:
        return frame
    try:
        bounds = grid.get_cell_bounds(conflict_cell) if conflict_cell else None
    except Exception:
        bounds = None
    if bounds is None:
        return frame

    x1, y1, x2, y2 = bounds
    cs = int(grid.cell_size)
    r = radius_cells * cs

    roi_x1 = max(int(grid.x_min), x1 - r)
    roi_x2 = min(int(grid.x_max), x2 + r)
    roi_y1 = max(int(grid.y_min), y1 - r)
    roi_y2 = min(int(grid.y_max), y2 + r)

    overlay = frame.copy()
    step = cs * max(1, skip)

    x_start = int(grid.x_min) + ((roi_x1 - int(grid.x_min)) // step) * step
    for x in range(x_start, roi_x2 + 1, step):
        cv2.line(overlay, (x, roi_y1), (x, roi_y2), _COLOR_GRID_LINE, _GRID_LINE_THICKNESS)
    y_start = int(grid.y_min) + ((roi_y1 - int(grid.y_min)) // step) * step
    for y in range(y_start, roi_y2 + 1, step):
        cv2.line(overlay, (roi_x1, y), (roi_x2, y), _COLOR_GRID_LINE, _GRID_LINE_THICKNESS)

    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def _draw_cell_highlight(
    frame: np.ndarray,
    grid,
    cell_id: str | None,
    *,
    alpha: float = 0.18,
    border_thickness: int = 3,
) -> np.ndarray:
    """Draw a solid cyan border + light fill on the conflict cell."""
    if grid is None or not cell_id or cell_id == "unknown":
        return frame
    try:
        bounds = grid.get_cell_bounds(cell_id)
    except Exception:
        return frame
    if bounds is None:
        return frame
    x1, y1, x2, y2 = bounds
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), _COLOR_GRID_CELL, -1)
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), _COLOR_GRID_CELL, border_thickness)
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


def _draw_arrowhead(
    frame: np.ndarray,
    p_from: tuple[int, int],
    p_to: tuple[int, int],
    color: tuple[int, int, int],
    size: int = 14,
) -> None:
    """Draw a triangular arrowhead at p_to, pointing in direction p_from→p_to."""
    dx, dy = p_to[0] - p_from[0], p_to[1] - p_from[1]
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1:
        return
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    tip = p_to
    base_x = tip[0] - ux * size
    base_y = tip[1] - uy * size
    left = (int(base_x + px * size * 0.6), int(base_y + py * size * 0.6))
    right = (int(base_x - px * size * 0.6), int(base_y - py * size * 0.6))
    pts = np.array([(int(tip[0]), int(tip[1])), left, right], dtype=np.int32)
    cv2.fillPoly(frame, [pts], color)


def _draw_polyline_with_arrow(
    frame: np.ndarray,
    points: list[tuple[float, float]],
    color: tuple[int, int, int],
    max_gap_px: float = 250.0,
) -> None:
    """Draw polyline but break at large jumps (tracker ID switches)."""
    pts = [(int(p[0]), int(p[1])) for p in points]
    if len(pts) < 2:
        return
    for i in range(len(pts) - 1):
        dx = pts[i + 1][0] - pts[i][0]
        dy = pts[i + 1][1] - pts[i][1]
        if (dx * dx + dy * dy) ** 0.5 > max_gap_px:
            continue
        cv2.line(frame, pts[i], pts[i + 1], color, 2)
    dx = pts[-1][0] - pts[-2][0]
    dy = pts[-1][1] - pts[-2][1]
    if (dx * dx + dy * dy) ** 0.5 <= max_gap_px:
        _draw_arrowhead(frame, pts[-2], pts[-1], color)


def _draw_badge_label(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> tuple[int, int, int, int]:
    """Draw text with a filled colored badge background. Returns box (x1,y1,x2,y2)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, _LABEL_SCALE, _LABEL_THICKNESS)
    pad = 4
    x1, y1 = x, y - th - 2 * pad
    x2, y2 = x + tw + 2 * pad, y
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    cv2.putText(frame, text, (x + pad, y - pad), font, _LABEL_SCALE, _COLOR_WHITE, _LABEL_THICKNESS)
    return (x1, y1, x2, y2)


def _draw_bbox_with_label(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: tuple[int, int, int],
    label: str,
    existing_labels: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int]:
    """Draw bbox + label above it, avoiding overlap with existing labels."""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(label, font, _LABEL_SCALE, _LABEL_THICKNESS)
    pad = 4
    box_h = th + 2 * pad

    # Try candidate positions, avoiding overlap
    candidates = [
        (x1, y1 - 2),
        (x1, y1 - box_h - 4),
        (x2 - tw - 2 * pad, y1 - 2),
        (x2 - tw - 2 * pad, y1 - box_h - 4),
        (x1, y2 + box_h + 2),
        (x2 - tw - 2 * pad, y2 + box_h + 2),
    ]
    chosen_x = x1
    chosen_y = y1 - 2
    for cx, cy in candidates:
        proposed = (cx, cy - box_h, cx + tw + 2 * pad, cy)
        overlaps = any(
            not (
                proposed[2] < ex[0]
                or proposed[0] > ex[2]
                or proposed[3] < ex[1]
                or proposed[1] > ex[3]
            )
            for ex in existing_labels
        )
        if not overlaps:
            chosen_x = cx
            chosen_y = cy
            break
    return _draw_badge_label(frame, label, chosen_x, chosen_y, color)


def _place_ring_pair(
    a_pos: tuple[int, int] | None,
    b_pos: tuple[int, int] | None,
    min_separation: float = 30.0,
) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    """If two ring centers are within min_separation, push apart vertically."""
    if a_pos is None or b_pos is None:
        return a_pos, b_pos
    dx = a_pos[0] - b_pos[0]
    dy = a_pos[1] - b_pos[1]
    dist = (dx * dx + dy * dy) ** 0.5
    if dist >= min_separation:
        return a_pos, b_pos
    offset = int((min_separation - dist) / 2 + 6)
    return (a_pos[0], a_pos[1] - offset), (b_pos[0], b_pos[1] + offset)


def _draw_frame_point_label(
    frame: np.ndarray,
    traj: list[tuple[float, float, float]],
    target_frame: int | None,
    color: tuple[int, int, int],
    label: str,
    existing_labels: list[tuple[int, int, int, int]],
) -> bool:
    if target_frame is None:
        return False
    for p in traj:
        if int(p[2]) == int(target_frame):
            x, y = int(p[0]), int(p[1])
            cv2.circle(frame, (x, y), 12, color, 3)
            cv2.circle(frame, (x, y), 4, color, -1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label, font, _LABEL_SCALE, _LABEL_THICKNESS)
            pad = 4
            box_h = th + 2 * pad
            h_img, w_img = frame.shape[:2]
            candidates = [
                (x + 16, y + 5),
                (x - tw - 2 * pad - 16, y + 5),
                (x + 16, y - box_h),
                (x - tw - 2 * pad - 16, y - box_h),
                (x + 16, y + box_h + 16),
                (x - tw - 2 * pad - 16, y + box_h + 16),
            ]
            chosen = (x + 16, y + 5)
            for cx, cy in candidates:
                if cx < 0 or cx + tw + 2 * pad > w_img or cy < box_h or cy > h_img:
                    continue
                proposed = (cx, cy - box_h, cx + tw + 2 * pad, cy)
                overlaps = any(
                    not (
                        proposed[2] < ex[0]
                        or proposed[0] > ex[2]
                        or proposed[3] < ex[1]
                        or proposed[1] > ex[3]
                    )
                    for ex in existing_labels
                )
                if not overlaps:
                    chosen = (cx, cy)
                    break
            _draw_badge_label(frame, label, chosen[0], chosen[1], color)
            return True
    return False


# ══════════════════════════════════════════════════════════════════════════
# HUD panel
# ══════════════════════════════════════════════════════════════════════════


def _draw_hud_panel(
    frame: np.ndarray,
    title: str,
    sections: list[tuple[str, list[str]]],
) -> np.ndarray:
    """Rounded semi-transparent HUD with title, sections (label → lines)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_scale = 0.72
    label_scale = 0.58
    body_scale = 0.55
    pad = 18
    section_gap = 6
    line_h = 22

    # Measure
    (title_w, title_h), _ = cv2.getTextSize(title, font, title_scale, 2)
    max_w = title_w
    total_h = pad + title_h + 10
    for sec_label, sec_lines in sections:
        (lw, lh), _ = cv2.getTextSize(sec_label, font, label_scale, 1)
        max_w = max(max_w, lw)
        total_h += section_gap + lh
        for line in sec_lines:
            (bw, _), _ = cv2.getTextSize(line, font, body_scale, 1)
            max_w = max(max_w, bw)
            total_h += line_h
    panel_w = max_w + 2 * pad
    panel_h = total_h + pad

    x0, y0 = 14, 14
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), _COLOR_PANEL_BG, -1)
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), _COLOR_PANEL_BORDER, 1)
    frame = cv2.addWeighted(overlay, 0.82, frame, 0.18, 0)

    # Title
    ty = y0 + pad + title_h
    cv2.putText(frame, title, (x0 + pad, ty), font, title_scale, _COLOR_PANEL_TITLE, 2)
    cy = ty + 10

    for sec_label, sec_lines in sections:
        cy += section_gap
        cv2.putText(
            frame,
            sec_label,
            (int(x0 + pad), int(cy + label_scale * 18)),
            font,
            label_scale,
            _COLOR_PANEL_BORDER,
            1,
        )
        cy += int(label_scale * 18) + 4
        for line in sec_lines:
            cv2.putText(
                frame,
                line,
                (int(x0 + pad + 6), int(cy + body_scale * 18)),
                font,
                body_scale,
                _COLOR_PANEL_TEXT,
                1,
            )
            cy += line_h
    return frame


# ══════════════════════════════════════════════════════════════════════════
# Main per-event renderer
# ══════════════════════════════════════════════════════════════════════════


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
    a_entry_frame: int | None = None,
    b_exit_frame: int | None = None,
    a_entry_time: float | None = None,
    b_exit_time: float | None = None,
) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None

    # 1. Grid (localized, dim) — then cell highlight on top
    frame = _draw_localized_grid(frame, grid, grid_cell)
    frame = _draw_cell_highlight(frame, grid, grid_cell)

    # 2. Trajectories with arrowheads
    if len(traj_a) >= 2:
        _draw_polyline_with_arrow(frame, [(p[0], p[1]) for p in traj_a], _COLOR_A)
    if len(traj_b) >= 2:
        _draw_polyline_with_arrow(frame, [(p[0], p[1]) for p in traj_b], _COLOR_B)

    # 3. Intersection ring (thicker)
    ip = _find_intersection_point(traj_a, traj_b)
    if ip:
        cx, cy = int(ip[0]), int(ip[1])
        cv2.circle(frame, (cx, cy), 18, _COLOR_MARK, 4)
        cv2.circle(frame, (cx, cy), 6, _COLOR_MARK, -1)

    # 4. Bounding boxes + collision-resolved labels
    existing_labels: list[tuple[int, int, int, int]] = []
    if bbox_a:
        bb = _draw_bbox_with_label(frame, bbox_a, _COLOR_A, f"T{track_a}", existing_labels)
        existing_labels.append(bb)
    if bbox_b:
        bb = _draw_bbox_with_label(frame, bbox_b, _COLOR_B, f"T{track_b}", existing_labels)
        existing_labels.append(bb)

    # 5. Entry/exit markers with ring NMS
    a_pt = b_pt = None
    if a_exit_frame is not None:
        for p in traj_a:
            if int(p[2]) == int(a_exit_frame):
                a_pt = (int(p[0]), int(p[1]))
                break
    if b_entry_frame is not None:
        for p in traj_b:
            if int(p[2]) == int(b_entry_frame):
                b_pt = (int(p[0]), int(p[1]))
                break
    a_pt, b_pt = _place_ring_pair(a_pt, b_pt, min_separation=30.0)

    a_label = f"A out f={a_exit_frame}" if a_exit_frame is not None else "A out f=?"
    b_label = f"B in f={b_entry_frame}" if b_entry_frame is not None else "B in f=?"
    if a_pt:
        cv2.circle(frame, a_pt, 12, _COLOR_A, 3)
        cv2.circle(frame, a_pt, 4, _COLOR_A, -1)
        _draw_badge_label(frame, a_label, a_pt[0] + 16, a_pt[1] + 5, _COLOR_A)
    if b_pt:
        cv2.circle(frame, b_pt, 12, _COLOR_B, 3)
        cv2.circle(frame, b_pt, 4, _COLOR_B, -1)
        _draw_badge_label(frame, b_label, b_pt[0] + 16, b_pt[1] + 5, _COLOR_B)

    # 6. HUD panel
    title = f"PET Sample {event_idx + 1}/{total_events}  ·  CSV row {csv_row}"
    timing_lines = [f"PET = {pet_val:.3f}s"]
    # Determine which branch computed PET and show the correct formula.
    # Branch A (standard): A clears first, then B enters → (B_entry - A_exit)
    # Branch B: B clears first, then A enters → (A_entry - B_exit)
    shown_arith = False
    if (
        a_exit_frame is not None
        and b_entry_frame is not None
        and a_exit_time is not None
        and b_entry_time is not None
        and int(a_exit_frame) < int(b_entry_frame)
    ):
        frames = int(b_entry_frame) - int(a_exit_frame)
        timing_lines.append(f"  = B_in({b_entry_time:.2f}s) − A_out({a_exit_time:.2f}s)")
        timing_lines.append(f"  = f{b_entry_frame} − f{a_exit_frame}  ({frames}f)")
        shown_arith = True
    elif (
        a_entry_frame is not None
        and b_exit_frame is not None
        and a_entry_time is not None
        and b_exit_time is not None
        and int(b_exit_frame) < int(a_entry_frame)
    ):
        frames = int(a_entry_frame) - int(b_exit_frame)
        timing_lines.append(f"  = A_in({a_entry_time:.2f}s) − B_out({b_exit_time:.2f}s)")
        timing_lines.append(f"  = f{a_entry_frame} − f{b_exit_frame}  ({frames}f)")
        shown_arith = True
    if not shown_arith:
        a_out_s = f"f={a_exit_frame}" if a_exit_frame is not None else "?"
        b_in_s = f"f={b_entry_frame}" if b_entry_frame is not None else "?"
        timing_lines.append(f"  A_out {a_out_s}, B_in {b_in_s}")
        timing_lines.append("  (see CSV for branch)")

    metadata_lines = [
        f"Frame: {frame_idx}",
        f"Grid:  {grid_cell}",
        f"Type:  {conflict_type}",
        f"Pair:  T{track_a} ↔ T{track_b}",
        f"A out: f={a_exit_frame if a_exit_frame is not None else '?'}",
        f"B in:  f={b_entry_frame if b_entry_frame is not None else '?'}",
    ]
    frame = _draw_hud_panel(
        frame,
        title,
        [
            ("TIMING", timing_lines),
            ("CONTEXT", metadata_lines),
        ],
    )
    return frame


# ══════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════


def render_pet_samples(
    events_csv: str | Path,
    detections_csv: str | Path,
    video_path: str | Path,
    out_dir: str | Path,
    n: int = 5,
    grid_config_path: str | Path | None = None,
) -> list[Path]:
    """Render the N lowest-PET events as annotated JPG samples."""
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
        .reset_index()
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

        def _gi(v):
            return None if pd.isna(v) else int(v)

        def _gf(v):
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
            a_exit_frame=_gi(row.get("track_a_exit_frame")),
            b_entry_frame=_gi(row.get("track_b_entry_frame")),
            a_exit_time=_gf(row.get("track_a_exit_time_sec")),
            b_entry_time=_gf(row.get("track_b_entry_time_sec")),
            a_entry_frame=_gi(row.get("track_a_entry_frame")),
            b_exit_frame=_gi(row.get("track_b_exit_frame")),
            a_entry_time=_gf(row.get("track_a_entry_time_sec")),
            b_exit_time=_gf(row.get("track_b_exit_time_sec")),
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
