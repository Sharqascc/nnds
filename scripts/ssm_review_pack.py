#!/usr/bin/env python3
"""Build a per-event review pack for the PET events.

For each PET event:
  - Extract a strip of N frames from the video across the event window
  - Draw the two interacting vehicles (A = red, B = blue) using the per-frame
    pixel coordinates in traj_a_json / traj_b_json and nearest-detection
    matching (the PET global track IDs do not appear in the detection CSV).
  - Compute geometric sanity checks
  - Emit a suggestion: likely_real / likely_false / ambiguous

Writes:
  - review_strips/event_XXX.jpg   (one strip per event)
  - gt_ssm_prelabel.csv            (suggestions + empty verdict columns)
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def _to_int(x, default=-1):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return default


def _parse_traj(raw) -> list:
    """Parse a JSON or Python-literal trajectory string."""
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


def _min_world_distance(traj_i, traj_j) -> float:
    if not traj_i or not traj_j:
        return float("inf")
    di = {p["frame"]: (p["world_x"], p["world_y"]) for p in traj_i if "world_x" in p}
    dj = {p["frame"]: (p["world_x"], p["world_y"]) for p in traj_j if "world_x" in p}
    common = set(di) & set(dj)
    if not common:
        # fall back to cross-frame min
        all_i = list(di.values())
        all_j = list(dj.values())
        if not all_i or not all_j:
            return float("inf")
        return float(min(np.hypot(ai[0] - bj[0], ai[1] - bj[1]) for ai in all_i for bj in all_j))
    return float(min(np.hypot(di[f][0] - dj[f][0], di[f][1] - dj[f][1]) for f in common))


def _traj_length(traj) -> int:
    return len(traj) if traj else 0


def _sanity(event_row) -> dict:
    traj_i = _parse_traj(event_row.get("traj_a_json"))
    traj_j = _parse_traj(event_row.get("traj_b_json"))

    min_dist = _min_world_distance(traj_i, traj_j)
    len_a = _traj_length(traj_i)
    len_b = _traj_length(traj_j)

    a_exit = event_row.get("track_a_exit_time_sec")
    b_entry = event_row.get("track_b_entry_time_sec")
    valid_order = False
    if pd.notna(a_exit) and pd.notna(b_entry):
        try:
            valid_order = float(a_exit) <= float(b_entry)
        except (TypeError, ValueError):
            valid_order = False

    pet_val = float(event_row["pet"]) if pd.notna(event_row["pet"]) else 0.0

    return {
        "min_world_dist_m": round(min_dist, 3) if np.isfinite(min_dist) else -1.0,
        "track_a_len": len_a,
        "track_b_len": len_b,
        "valid_pet_order": bool(valid_order),
        "pet_s": round(pet_val, 4),
    }


def _suggest(sanity: dict) -> tuple[str, str]:
    min_d = sanity["min_world_dist_m"]
    len_a = sanity["track_a_len"]
    len_b = sanity["track_b_len"]
    valid = sanity["valid_pet_order"]

    if len_a < 5 or len_b < 5:
        return "likely_false", f"track too short (a={len_a}, b={len_b})"
    if not valid:
        return "ambiguous", "A does not exit before B enters (overlap)"
    if min_d < 0:
        return "ambiguous", "no overlapping world frames between tracks"
    if min_d > 5.0:
        return "likely_false", f"world-space min distance {min_d} m > 5 m"
    if 0.05 <= min_d <= 3.0:
        return "likely_real", f"close approach {min_d} m, valid order"
    return "ambiguous", f"min_dist={min_d}, valid_order={valid}"


def _nearest_det(det_df: pd.DataFrame, frame: int, cx: float, cy: float, max_px: float = 120.0):
    frame_dets = det_df[det_df["frame"] == frame]
    if frame_dets.empty:
        return None
    best = None
    best_d2 = float("inf")
    for _, d in frame_dets.iterrows():
        dx = float(d["cx"]) - cx
        dy = float(d["cy"]) - cy
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best = d
    return best if best_d2 < max_px * max_px else None


def _draw_track(frame, det_df, frame_idx, traj_by_f, color, label):
    p = traj_by_f.get(frame_idx)
    if p is None:
        return
    cx = float(p["x_pixel"])
    cy = float(p["y_pixel"])
    d = _nearest_det(det_df, frame_idx, cx, cy)
    if d is not None:
        x1, y1, x2, y2 = int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
        cv2.putText(frame, label, (x1, max(30, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    else:
        cv2.drawMarker(frame, (int(cx), int(cy)), color, cv2.MARKER_CROSS, 30, 3)
        cv2.putText(
            frame, label, (int(cx) + 12, int(cy) - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3
        )


def _review_strip(
    cap: cv2.VideoCapture,
    event_row,
    det_df: pd.DataFrame,
    n_frames: int = 6,
    strip_width: int = 400,
) -> np.ndarray:
    traj_a = _parse_traj(event_row.get("traj_a_json"))
    traj_b = _parse_traj(event_row.get("traj_b_json"))

    a_entry = _to_int(event_row.get("track_a_entry_frame"), 0)
    a_exit = _to_int(event_row.get("track_a_exit_frame"), a_entry + 1)
    b_entry = _to_int(event_row.get("track_b_entry_frame"), a_entry)
    b_exit = _to_int(event_row.get("track_b_exit_frame"), a_exit)

    start = max(0, min(a_entry, b_entry) - 5)
    end = max(start + 1, max(a_exit, b_exit) + 5)
    frames = np.linspace(start, end - 1, n_frames).astype(int).tolist()

    a_by_f = {int(p["frame"]): p for p in traj_a if "x_pixel" in p}
    b_by_f = {int(p["frame"]): p for p in traj_b if "x_pixel" in p}

    tiles = []
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((720, 1600, 3), dtype=np.uint8)
        fh, fw = frame.shape[:2]

        _draw_track(frame, det_df, f, a_by_f, (0, 0, 255), "A")  # red
        _draw_track(frame, det_df, f, b_by_f, (255, 0, 0), "B")  # blue

        cv2.putText(frame, f"f={f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        scale = strip_width / fw
        tile = cv2.resize(frame, (strip_width, int(fh * scale)))
        tiles.append(tile)

    return cv2.hconcat(tiles) if tiles else np.zeros((100, strip_width, 3), dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--pet-csv", required=True)
    parser.add_argument("--detections-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-events", type=int, default=None)
    args = parser.parse_args()

    out = Path(args.out_dir)
    strips_dir = out / "review_strips"
    strips_dir.mkdir(parents=True, exist_ok=True)

    pet = pd.read_csv(args.pet_csv)
    if args.max_events:
        pet = pet.head(args.max_events)

    det = pd.read_csv(args.detections_csv)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {args.video}")

    rows = []
    for i, (_, ev) in enumerate(pet.iterrows()):
        sanity = _sanity(ev)
        suggestion, reason = _suggest(sanity)
        strip = _review_strip(cap, ev, det)
        strip_path = strips_dir / f"event_{i:03d}.jpg"
        cv2.imwrite(str(strip_path), strip, [cv2.IMWRITE_JPEG_QUALITY, 85])
        rows.append(
            {
                "event_idx": i,
                "track_a": _to_int(ev["track_a"], -1),
                "track_b": _to_int(ev["track_b"], -1),
                "pet": round(float(ev["pet"]), 4),
                "min_world_dist_m": sanity["min_world_dist_m"],
                "track_a_len": sanity["track_a_len"],
                "track_b_len": sanity["track_b_len"],
                "valid_pet_order": sanity["valid_pet_order"],
                "suggestion": suggestion,
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
    print("Suggestion distribution:")
    print(df["suggestion"].value_counts().to_string())


if __name__ == "__main__":
    main()
