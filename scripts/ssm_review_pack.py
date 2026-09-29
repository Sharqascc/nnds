#!/usr/bin/env python3
"""Build per-event review strips for PET events.

Draws on each frame:
  - faint grid lines
  - the event's cell highlighted (translucent yellow + border)
  - smoothed trajectory trails that ride the ground contact corner
    (A = leader -> bottom-trailing corner; B = follower -> bottom-leading corner)
  - bounding boxes with small A / B chips and a matching ground marker
Frame layout is a 2x3 tile grid with a header bar carrying cell + metadata.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.grid_overlay import (
    cell_bounds_px,
    draw_grid_lines,
    grid_from_config,
    parse_cell_name,
)


def _to_int(x, default=-1):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return default


def _parse_traj(raw) -> list:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if isinstance(raw, list):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        return json.loads(raw)
    except Exception:
        pass
    try:
        v = ast.literal_eval(raw)
        return v if isinstance(v, list) else []
    except Exception:
        return []


def _min_world_distance(ti, tj) -> float:
    if not ti or not tj:
        return float("inf")
    di = {p["frame"]: (p["world_x"], p["world_y"]) for p in ti if "world_x" in p}
    dj = {p["frame"]: (p["world_x"], p["world_y"]) for p in tj if "world_x" in p}
    common = set(di) & set(dj)
    if not common:
        ai = list(di.values())
        bj = list(dj.values())
        if not ai or not bj:
            return float("inf")
        return float(min(np.hypot(a[0] - b[0], a[1] - b[1]) for a in ai for b in bj))
    return float(min(np.hypot(di[f][0] - dj[f][0], di[f][1] - dj[f][1]) for f in common))


def _sanity(ev) -> dict:
    ti = _parse_traj(ev.get("traj_a_json"))
    tj = _parse_traj(ev.get("traj_b_json"))
    md = _min_world_distance(ti, tj)
    a_exit = ev.get("track_a_exit_time_sec")
    b_entry = ev.get("track_b_entry_time_sec")
    valid = False
    if pd.notna(a_exit) and pd.notna(b_entry):
        try:
            valid = float(a_exit) <= float(b_entry)
        except (TypeError, ValueError):
            valid = False
    return {
        "min_world_dist_m": round(md, 3) if np.isfinite(md) else -1.0,
        "track_a_len": len(ti),
        "track_b_len": len(tj),
        "valid_pet_order": bool(valid),
    }


def _suggest(s: dict) -> tuple[str, str]:
    md = s["min_world_dist_m"]
    if s["track_a_len"] < 5 or s["track_b_len"] < 5:
        return "likely_false", f"short tracks a={s['track_a_len']} b={s['track_b_len']}"
    if not s["valid_pet_order"]:
        return "ambiguous", "A does not exit before B enters (overlap)"
    if md < 0:
        return "ambiguous", "no overlapping world frames"
    if md > 5.0:
        return "likely_false", f"world min dist {md} m > 5 m"
    if 0.05 <= md <= 3.0:
        return "likely_real", f"close approach {md} m, valid order"
    return "ambiguous", f"min_dist={md}"


def _det_index(det_df: pd.DataFrame) -> dict[int, list]:
    frames = pd.to_numeric(det_df["frame"], errors="coerce").astype(int).to_numpy()
    cxs = pd.to_numeric(det_df["cx"], errors="coerce").astype(float).to_numpy()
    cys = pd.to_numeric(det_df["cy"], errors="coerce").astype(float).to_numpy()
    x1s = pd.to_numeric(det_df["x1"], errors="coerce").astype(int).to_numpy()
    y1s = pd.to_numeric(det_df["y1"], errors="coerce").astype(int).to_numpy()
    x2s = pd.to_numeric(det_df["x2"], errors="coerce").astype(int).to_numpy()
    y2s = pd.to_numeric(det_df["y2"], errors="coerce").astype(int).to_numpy()
    out: dict[int, list] = {}
    for i in range(len(det_df)):
        out.setdefault(int(frames[i]), []).append(
            (float(cxs[i]), float(cys[i]), int(x1s[i]), int(y1s[i]), int(x2s[i]), int(y2s[i]))
        )
    return out


def _nearest_det(dets: list, cx: float, cy: float, max_px: float = 120.0):
    best = None
    best_d2 = max_px * max_px
    for d in dets:
        dx = d[0] - cx
        dy = d[1] - cy
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best = d
    return best


def _sorted_traj(pts: list) -> list[tuple[int, float, float]]:
    return sorted(
        [
            (int(p["frame"]), float(p["x_pixel"]), float(p["y_pixel"]))
            for p in pts
            if "x_pixel" in p
        ],
        key=lambda t: t[0],
    )


def _smooth(xs: np.ndarray, ys: np.ndarray, window: int = 7):
    if window < 3 or len(xs) < window:
        return xs, ys
    if window % 2 == 0:
        window += 1
    k = np.ones(window) / window
    pad = window // 2
    xs_p = np.concatenate([np.full(pad, xs[0]), xs, np.full(pad, xs[-1])])
    ys_p = np.concatenate([np.full(pad, ys[0]), ys, np.full(pad, ys[-1])])
    return np.convolve(xs_p, k, mode="valid"), np.convolve(ys_p, k, mode="valid")


def _heading_at(traj_sorted, frame: int) -> tuple[float, float]:
    n = len(traj_sorted)
    if n < 2:
        return 0.0, 0.0
    idx = min(range(n), key=lambda i: abs(traj_sorted[i][0] - frame))
    if idx == 0:
        _, x0, y0 = traj_sorted[0]
        _, x1, y1 = traj_sorted[1]
        return x1 - x0, y1 - y0
    if idx == n - 1:
        _, x0, y0 = traj_sorted[-2]
        _, x1, y1 = traj_sorted[-1]
        return x1 - x0, y1 - y0
    _, x0, y0 = traj_sorted[idx - 1]
    _, x1, y1 = traj_sorted[idx + 1]
    return x1 - x0, y1 - y0


def _ground_point(box, heading_dx: float, role: str) -> tuple[int, int]:
    """Return the ground-contact marker for a box.

    leader: trailing bottom corner (last point to leave the zone)
    follower: leading bottom corner (first point to enter the zone)
    """
    x1, _, x2, y2 = box
    cx = (x1 + x2) // 2
    by = y2
    if abs(heading_dx) >= 1.5:
        if heading_dx > 0:
            front_x, rear_x = x2, x1
        else:
            front_x, rear_x = x1, x2
    else:
        front_x = rear_x = cx
    return (int(rear_x), int(by)) if role == "leader" else (int(front_x), int(by))


def _ground_trail_points(traj_sorted, det_index, role: str) -> list[tuple[int, int, int]]:
    """Bottom-center of each detection box (cx, y2) — the ground-contact point."""
    _ = role
    out: list[tuple[int, int, int]] = []
    for f, cx, cy in traj_sorted:
        dets = det_index.get(f, [])
        d = _nearest_det(dets, cx, cy)
        if d is None:
            out.append((int(f), int(cx), int(cy)))
            continue
        _, _, x1, _, x2, y2 = d
        bx = (int(x1) + int(x2)) // 2
        by = int(y2)
        out.append((int(f), bx, by))
    return out


def _draw_trail_smooth(frame, ground_points, color, thickness=2):
    if len(ground_points) < 2:
        return
    xs = np.array([p[1] for p in ground_points], dtype=float)
    ys = np.array([p[2] for p in ground_points], dtype=float)
    xs_s, ys_s = _smooth(xs, ys, window=7)
    poly = np.stack([xs_s, ys_s], axis=1).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(
        frame, [poly], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA
    )


def _draw_box_marker(frame, dets_at_frame, cx, cy, color, label, role, heading_dx):
    """Circle is placed at the trajectory point (box center, cx/cy) so it
    matches the pipeline's grid_cell computation."""
    _ = role, heading_dx
    d = _nearest_det(dets_at_frame, cx, cy)
    if d is None:
        return
    _, _, x1, y1, x2, y2 = d
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (int(x1), int(y1) - th - 8), (int(x1) + tw + 8, int(y1)), color, -1)
    cv2.putText(
        frame,
        label,
        (int(x1) + 4, int(y1) - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    tx, ty = int(cx), int(cy)
    cv2.circle(frame, (tx, ty), 7, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, (tx, ty), 5, color, -1, cv2.LINE_AA)


def _draw_cell_overlay(frame, cell_name, dims, color=(0, 255, 255), alpha=0.22, border_thickness=3):
    if not isinstance(cell_name, str) or not cell_name:
        return frame
    parsed = parse_cell_name(cell_name, dims.naming_style)
    if parsed is None:
        return frame
    col, row = parsed
    x1, y1, x2, y2 = cell_bounds_px(col, row, dims)
    h, w = frame.shape[:2]
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return frame
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, border_thickness, cv2.LINE_AA)
    return frame


def _review_strip(cap, ev, det_index, grid_dims=None, n_frames=6, tile_width=700):
    ta = _sorted_traj(_parse_traj(ev.get("traj_a_json")))
    tb = _sorted_traj(_parse_traj(ev.get("traj_b_json")))

    a_entry = _to_int(ev.get("track_a_entry_frame"), 0)
    a_exit = _to_int(ev.get("track_a_exit_frame"), a_entry + 1)
    b_entry = _to_int(ev.get("track_b_entry_frame"), a_entry)
    b_exit = _to_int(ev.get("track_b_exit_frame"), a_exit)

    start = max(0, min(a_entry, b_entry) - 5)
    end = max(start + 1, max(a_exit, b_exit) + 5)
    frames = np.linspace(start, end - 1, n_frames).astype(int).tolist()

    cell_txt = ev.get("grid_cell")
    ta_ground = _ground_trail_points(ta, det_index, "leader")
    tb_ground = _ground_trail_points(tb, det_index, "follower")

    tiles = []
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((720, 1600, 3), dtype=np.uint8)
        fh, fw = frame.shape[:2]

        if grid_dims is not None:
            frame = draw_grid_lines(frame, grid_dims, color=(110, 110, 110), thickness=1)
            frame = _draw_cell_overlay(frame, cell_txt, grid_dims)

        _draw_trail_smooth(frame, ta_ground, (0, 0, 255), thickness=2)
        _draw_trail_smooth(frame, tb_ground, (255, 100, 0), thickness=2)

        dets_here = det_index.get(int(f), [])
        a_pt = next(((x, y) for fr, x, y in ta if fr == f), None)
        b_pt = next(((x, y) for fr, x, y in tb if fr == f), None)
        if a_pt is not None:
            _draw_box_marker(
                frame,
                dets_here,
                a_pt[0],
                a_pt[1],
                (0, 0, 255),
                "A",
                "leader",
                _heading_at(ta, f)[0],
            )
        if b_pt is not None:
            _draw_box_marker(
                frame,
                dets_here,
                b_pt[0],
                b_pt[1],
                (255, 100, 0),
                "B",
                "follower",
                _heading_at(tb, f)[0],
            )

        cv2.putText(
            frame, f"f={f}", (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA
        )

        scale = tile_width / fw
        tile = cv2.resize(frame, (tile_width, int(fh * scale)), interpolation=cv2.INTER_AREA)
        tiles.append(tile)

    while len(tiles) < 6:
        tiles.append(np.zeros_like(tiles[0]))
    grid_img = cv2.vconcat([cv2.hconcat(tiles[:3]), cv2.hconcat(tiles[3:6])])

    header_h = 64
    header = np.zeros((header_h, grid_img.shape[1], 3), dtype=np.uint8)
    a_exit = ev.get("track_a_exit_time_sec")
    b_entry = ev.get("track_b_entry_time_sec")
    pet_val = float(ev.get("pet", 0))
    a_s = f"{float(a_exit):.3f} s" if pd.notna(a_exit) else "?"
    b_s = f"{float(b_entry):.3f} s" if pd.notna(b_entry) else "?"
    hdr = (
        f"{cell_txt}   |   A={_to_int(ev.get('track_a'), -1)} exits @ {a_s}   "
        f"B={_to_int(ev.get('track_b'), -1)} enters @ {b_s}   |   "
        f"PET = {pet_val:.3f} s"
    )
    cv2.putText(
        header, hdr, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA
    )
    return cv2.vconcat([header, grid_img])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--pet-csv", required=True)
    ap.add_argument("--detections-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--grid-config", default="configs/GITI_grid_config.json")
    args = ap.parse_args()

    out = Path(args.out_dir)
    strips_dir = out / "review_strips"
    strips_dir.mkdir(parents=True, exist_ok=True)

    grid_cfg = json.loads(Path(args.grid_config).read_text())
    grid_dims = grid_from_config(grid_cfg)

    pet = pd.read_csv(args.pet_csv)
    if args.max_events:
        pet = pet.head(args.max_events)
    det = pd.read_csv(args.detections_csv)
    det_index = _det_index(det)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {args.video}")

    rows = []
    for i, (_, ev) in enumerate(pet.iterrows()):
        s = _sanity(ev)
        sug, reason = _suggest(s)
        strip = _review_strip(cap, ev, det_index, grid_dims=grid_dims)
        cv2.imwrite(str(strips_dir / f"event_{i:03d}.jpg"), strip, [cv2.IMWRITE_JPEG_QUALITY, 88])
        rows.append(
            {
                "event_idx": i,
                "track_a": _to_int(ev["track_a"], -1),
                "track_b": _to_int(ev["track_b"], -1),
                "grid_cell": ev.get("grid_cell", ""),
                "pet": round(float(ev["pet"]), 4),
                "min_world_dist_m": s["min_world_dist_m"],
                "track_a_len": s["track_a_len"],
                "track_b_len": s["track_b_len"],
                "valid_pet_order": s["valid_pet_order"],
                "suggestion": sug,
                "reason": reason,
                "verdict": "",
                "actual_pet": "",
                "notes": "",
            }
        )

    cap.release()
    df = pd.DataFrame(rows)
    df.to_csv(out / "gt_ssm_prelabel.csv", index=False)

    print(f"Wrote {len(df)} review strips to {strips_dir}")
    print(f"Wrote {out / 'gt_ssm_prelabel.csv'}")
    print()
    print(df["suggestion"].value_counts().to_string())


if __name__ == "__main__":
    main()
