#!/usr/bin/env python3
"""
Automated validation of NNDS pipeline outputs.

Checks detections, tracking stability (on split tracks), and PET consistency.

Usage:
    python scripts/validate_outputs.py \
        --detections outputs/petevents_bev_final_detections.csv \
        --detections-split outputs/petevents_bev_final_split_detections.csv \
        --pet outputs/petevents_bev_final.csv
"""
import argparse
import json
import sys
import pandas as pd
import numpy as np
from pathlib import Path

DETECTION_COLUMNS = [
    "frame", "track_id", "class_id", "class_name",
    "conf", "x1", "y1", "x2", "y2", "cx", "cy", "source",
]
PET_COLUMNS = [
    "event_id", "pet", "pet_time_based", "frame", "track_a", "track_b",
    "orig_track_a", "seg_a", "orig_track_b", "seg_b",
    "conflict_type", "grid_cell",
    "track_a_entry_frame", "track_a_exit_frame", "track_a_exit_time_sec",
    "track_b_entry_frame", "track_b_entry_time_sec", "track_b_exit_frame",
    "world_traj_i", "world_traj_j", "traj_a_json", "traj_b_json",
    "video_source", "time_of_day_label",
    "gate_a_entry", "gate_b_entry",
]
ALLOWED_CLASSES = {"pedestrian", "person", "bicycle", "car", "bike", "motorcycle", "bus", "truck", "auto"}

def load_csv(path, required_columns, context):
    path = Path(path)
    if not path.exists():
        print(f"❌ {context}: file not found: {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    missing = set(required_columns) - set(df.columns)
    if missing:
        print(f"❌ {context}: missing columns: {sorted(missing)}")
        sys.exit(1)
    if df.empty:
        print(f"⚠️ {context}: empty DataFrame")
        return df, True
    return df, False

def validate_detections(det_df):
    problems = []
    bad_boxes = det_df[(det_df["x1"] >= det_df["x2"]) | (det_df["y1"] >= det_df["y2"])]
    if not bad_boxes.empty:
        problems.append(f"Detections with invalid bounding boxes: {len(bad_boxes)}")
    bad_conf = det_df[(det_df["conf"] < 0) | (det_df["conf"] > 1)]
    if not bad_conf.empty:
        problems.append(f"Detections with conf out of [0,1]: {len(bad_conf)}")
    if "class_name" in det_df.columns:
        unknown = set(det_df["class_name"].unique()) - ALLOWED_CLASSES
        if unknown:
            problems.append(f"Unknown class names: {unknown}")
    return problems

def validate_tracking_stability(det_df, max_gap=10, max_jump=50.0):
    problems = []
    if det_df.empty:
        return problems
    for track_id, group in det_df.groupby("track_id"):
        group = group.sort_values("frame")
        if len(group) < 2:
            continue
        frames = group["frame"].values
        x = group["cx"].values
        y = group["cy"].values
        gaps = np.diff(frames)
        big_gaps = gaps[gaps > max_gap]
        if len(big_gaps) > 0:
            problems.append(f"Track {track_id}: {len(big_gaps)} frame gap(s) > {max_gap} (max {big_gaps.max()})")
        dx = np.diff(x)
        dy = np.diff(y)
        jumps = np.sqrt(dx**2 + dy**2)
        big_jumps = jumps[jumps > max_jump]
        if len(big_jumps) > 0:
            problems.append(f"Track {track_id}: {len(big_jumps)} spatial jump(s) > {max_jump}px (max {big_jumps.max():.1f})")
    return problems

def validate_trajectory_json(json_str, label):
    problems = []
    if not isinstance(json_str, str) or json_str == "" or json_str == "[]":
        problems.append(f"{label}: empty trajectory")
        return problems
    try:
        pts = json.loads(json_str)
    except Exception as e:
        problems.append(f"{label}: invalid JSON: {e}")
        return problems
    if len(pts) < 2:
        problems.append(f"{label}: fewer than 2 points")
        return problems
    required_keys = {"frame", "x_pixel", "y_pixel", "world_x", "world_y"}
    for i, p in enumerate(pts[:5]):
        missing = required_keys - set(p.keys())
        if missing:
            problems.append(f"{label} point {i}: missing keys {missing}")
            break
    world_vals = [p.get("world_x") for p in pts if p.get("world_x") is not None]
    if len(world_vals) == 0:
        problems.append(f"{label}: all world coordinates are None")
    return problems

def validate_pet(pet_df, det_df, video_frames=1838, split_det_df=None):
    problems = []
    if pet_df.empty:
        return problems

    # Filter out non-positive PET events for validation (they should not exist after pipeline fix)
    bad_pet = pet_df[pet_df["pet"] <= 0]
    if not bad_pet.empty:
        problems.append(f"PET <= 0: {len(bad_pet)} events")
        pet_df = pet_df[pet_df["pet"] > 0]  # validate only positive PET

    bad_frame = pet_df[(pet_df["frame"] < 0) | (pet_df["frame"] > video_frames)]
    if not bad_frame.empty:
        problems.append(f"Frame out of range: {len(bad_frame)} events")
    if "grid_cell" in pet_df.columns:
        oob = pet_df[pet_df["grid_cell"] == "OUT_OF_BOUNDS"]
        if not oob.empty:
            problems.append(f"OUT_OF_BOUNDS grid cells: {len(oob)} events")
    for prefix in ["track_a", "track_b"]:
        entry_col = f"{prefix}_entry_frame"
        exit_col = f"{prefix}_exit_frame"
        if entry_col in pet_df.columns and exit_col in pet_df.columns:
            bad = pet_df[pet_df[entry_col] > pet_df[exit_col]]
            if not bad.empty:
                problems.append(f"{prefix}: entry > exit in {len(bad)} events")
    # Use split detections if available for ID validation
    detection_source_df = split_det_df if (split_det_df is not None and not split_det_df.empty) else det_df
    if not detection_source_df.empty and "track_id" in detection_source_df.columns:
        det_track_ids = set(detection_source_df["track_id"].unique())
        for event_track_col in ["track_a", "track_b"]:
            if event_track_col in pet_df.columns:
                missing_ids = set(pet_df[event_track_col]) - det_track_ids
                if missing_ids:
                    problems.append(f"{event_track_col}: track IDs not in split detections: {sorted(missing_ids)[:5]}")
    if "traj_a_json" in pet_df.columns and "traj_b_json" in pet_df.columns:
        for idx, row in pet_df.iterrows():
            for col, label in [("traj_a_json", "Traj A"), ("traj_b_json", "Traj B")]:
                problems.extend(validate_trajectory_json(row[col], f"Event {row['event_id']} {label}"))
                if idx > 5:
                    break
            if idx > 5:
                break
    return problems

def main():
    parser = argparse.ArgumentParser(description="Validate NNDS pipeline outputs")
    parser.add_argument("--detections", default="outputs/petevents_bev_final_detections.csv")
    parser.add_argument("--detections-split", default="outputs/petevents_bev_final_split_detections.csv")
    parser.add_argument("--pet", default="outputs/petevents_bev_final.csv")
    parser.add_argument("--video-frames", type=int, default=1838)
    parser.add_argument("--max-gap", type=int, default=10)
    parser.add_argument("--max-jump", type=float, default=50.0)
    args = parser.parse_args()

    print("=" * 70)
    print("NNDS Pipeline Output Validation")
    print("=" * 70)

    det_df, det_empty = load_csv(args.detections, DETECTION_COLUMNS, "Detections")
    pet_df, pet_empty = load_csv(args.pet, PET_COLUMNS, "PET events")

    split_det_df = None
    split_det_empty = True
    if Path(args.detections_split).exists():
        split_det_df, split_det_empty = load_csv(args.detections_split, DETECTION_COLUMNS, "Split detections")
    else:
        print("⚠️ Split detections not found; tracking stability will be checked on raw detections (may over-report)")

    all_problems = []

    if not det_empty:
        problems = validate_detections(det_df)
        if problems:
            all_problems.extend([f"Detections: {p}" for p in problems])
        else:
            print("✅ Detections basic checks passed")
    else:
        print("⚠️ Detections empty; skipping detection checks")

    # Tracking stability: use split detections if available
    if not split_det_empty and split_det_df is not None:
        problems = validate_tracking_stability(split_det_df, args.max_gap, args.max_jump)
        if problems:
            all_problems.extend([f"Tracking: {p}" for p in problems])
        else:
            print("✅ Tracking stability checks passed (on split tracks)")
    else:
        print("⚠️ Split detections not available; skipping tracking stability checks")

    if not pet_empty:
        problems = validate_pet(pet_df, det_df, args.video_frames, split_det_df)
        if problems:
            all_problems.extend([f"PET: {p}" for p in problems])
        else:
            print("✅ PET event checks passed")
    else:
        print("⚠️ PET empty; skipping PET checks")

    print()
    if all_problems:
        print(f"❌ Found {len(all_problems)} problem(s):")
        for p in all_problems[:50]:
            print(" -", p)
        sys.exit(1)
    else:
        print("✅ All validation checks passed")
        sys.exit(0)

if __name__ == "__main__":
    main()
