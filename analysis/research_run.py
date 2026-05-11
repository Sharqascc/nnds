#!/usr/bin/env python3
"""research_run.py - WORKING VERSION - Orchestrated NNDS research workflow"""


import argparse
import ast
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd

from analysis.visualization.video_overlays import VideoOverlayPlotter

ROOT = Path(__file__).resolve().parents[1]

def log(msg, level="INFO"):
    print(f"[{level}] {msg}")

def run_cmd(cmd: List[str], cwd: Path = ROOT) -> None:
    log(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        log(f"Command failed with return code {result.returncode}", "ERROR")
        raise SystemExit(result.returncode)


def _parse_worldsample_traj(value):
    if value is None:
        return []
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return []

    parts = [p.strip() for p in s.split(";") if p.strip()]
    traj = []
    ws_pattern = re.compile(r"WorldSample\(t=([-+0-9.eE]+),\s*x=([-+0-9.eE]+),\s*y=([-+0-9.eE]+)\)")

    for part in parts:
        m = ws_pattern.fullmatch(part)
        if m:
            traj.append((float(m.group(1)), float(m.group(2)), float(m.group(3))))
            continue
        try:
            item = ast.literal_eval(part)
            if isinstance(item, (tuple, list)) and len(item) >= 3:
                traj.append((float(item[0]), float(item[1]), float(item[2])))
        except Exception:
            continue
    return traj

def _world_to_bev_canvas(trajs, width=500, height=500, pad=30):
    points = []
    for traj in trajs:
        for _, x, y in traj:
            points.append((x, y))

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    if not points:
        return canvas

    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)

    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()

    if xmax - xmin < 1e-6:
        xmax += 1.0
    if ymax - ymin < 1e-6:
        ymax += 1.0

    def map_pt(x, y):
        u = pad + (x - xmin) / (xmax - xmin) * (width - 2 * pad)
        v = height - (pad + (y - ymin) / (ymax - ymin) * (height - 2 * pad))
        return int(round(u)), int(round(v))

    colors = [(178, 114, 0), (0, 159, 230), (115, 158, 0), (167, 121, 204)]

    for i, traj in enumerate(trajs):
        color = colors[i % len(colors)]
        pts = [map_pt(x, y) for _, x, y in traj]
        for j in range(len(pts) - 1):
            cv2.line(canvas, pts[j], pts[j + 1], color, 2)
        for pt in pts:
            cv2.circle(canvas, pt, 3, color, -1)

    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (0, 0, 0), 2)
    cv2.putText(canvas, "BEV", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    return canvas

def _export_overlay_videos(video_path: str, csv_path: str, fps: int = 30):
    df = pd.read_csv(csv_path)
    if df.empty:
        log("PET CSV is empty; skipping overlay video export.", "WARN")
        return

    event_row = None
    for _, row in df.iterrows():
        traj_i = _parse_worldsample_traj(row.get("world_traj_i"))
        traj_j = _parse_worldsample_traj(row.get("world_traj_j"))
        if traj_i and traj_j:
            event_row = row
            break

    if event_row is None:
        log("No event with non-empty world_traj_i/world_traj_j found; skipping overlay export.", "WARN")
        return

    traj_i = _parse_worldsample_traj(event_row.get("world_traj_i"))
    traj_j = _parse_worldsample_traj(event_row.get("world_traj_j"))
    track_ids = [
        int(event_row["track_a"]) if "track_a" in event_row and pd.notna(event_row["track_a"]) else 1,
        int(event_row["track_b"]) if "track_b" in event_row and pd.notna(event_row["track_b"]) else 2,
    ]

    frame_val = event_row["frame"] if "frame" in event_row else 0
    center_frame = int(frame_val) if pd.notna(frame_val) else 0
    start_frame = max(center_frame - 30, 0)
    end_frame = center_frame + 30

    pet_value = float(event_row["pet"]) if "pet" in event_row and pd.notna(event_row["pet"]) else None

    plotter = VideoOverlayPlotter(dpi=300)

    overlay_dir = Path("outputs/video_overlays")
    overlay_dir.mkdir(parents=True, exist_ok=True)

    analyzed_path = str(overlay_dir / "analyzed_video.mp4")
    bev_path = str(overlay_dir / "bev_overlay_video.mp4")

    plotter.generate_conflict_video(
        video_path=video_path,
        frame_range=(start_frame, end_frame),
        trajectories=[traj_i, traj_j],
        track_ids=track_ids,
        pet_value=pet_value,
        output_path=analyzed_path,
        fps=fps,
    )

    def inset_callback(frame_idx):
        return _world_to_bev_canvas([traj_i, traj_j])

    plotter.generate_conflict_video(
        video_path=video_path,
        frame_range=(start_frame, end_frame),
        trajectories=[traj_i, traj_j],
        track_ids=track_ids,
        pet_value=pet_value,
        output_path=bev_path,
        fps=fps,
        inset_callback=inset_callback,
        inset_position="bottom-right",
        inset_scale=0.28,
    )

    log(f"Overlay videos saved under {overlay_dir}", "SUCCESS")

def main():
    parser = argparse.ArgumentParser(description="NNDS Research Pipeline")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument(
        "--csv-path",
        help="Existing PET CSV to use for downstream analysis (skip Stage 1).",
    )
    parser.add_argument(
        "--vlm-annotate",
        action="store_true",
        help="Run VLM analysis on PET events and write *_vlm.csv.",
    )
    parser.add_argument("--sam3-weights", default="sam3.pt", help="SAM3 weights")
    parser.add_argument("--pet-threshold", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--train-diffusion", action="store_true")
    parser.add_argument("--eval-diffusion", action="store_true")
    parser.add_argument("--export-video", action="store_true",
                        help="Export analyzed conflict video with overlays.")
    parser.add_argument("--live-bev-overlay", action="store_true",
                        help="Export video with BEV inset overlay.")
    parser.add_argument("--run-all", action="store_true",
                        help="Run diffusion eval, analyzed video export, and BEV overlay export.")
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    if args.run_all:
        args.eval_diffusion = True
        args.export_video = True
        args.live_bev_overlay = True

    video_path = Path(args.video)
    if not video_path.exists() and not args.skip_extraction:
        log(f"Video not found: {video_path}", "ERROR")
        sys.exit(1)
    
    if not args.out_csv:
        stem = video_path.stem
        frame_tag = f"{args.max_frames}f" if args.max_frames else "full"
        pet_tag = str(args.pet_threshold).replace(".", "p")
        out_csv = f"outputs/petevents_bev_{stem}_{frame_tag}_pet{pet_tag}.csv"
    else:
        out_csv = args.out_csv
    
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    stages_completed = []
    
    if not args.skip_extraction:
        log("=== Stage 1: Video -> PET ===")
        cmd = [sys.executable, "traffic_analyzer.py", "--video", args.video,
               "--sam3-weights", args.sam3_weights, "--out-csv", out_csv,
               "--pet-threshold", str(args.pet_threshold)]
        if args.max_frames:
            cmd += ["--max-frames", str(args.max_frames)]
        if not args.dry_run:
            start = time.time()
            run_cmd(cmd, cwd=ROOT)
            log(f"Stage 1 completed in {time.time() - start:.1f}s", "SUCCESS")
            stages_completed.append("extraction")
    
    if args.train_diffusion:
        log("=== Stage 2: Diffusion Training ===")
        cmd = [sys.executable, "traffic_diffusion/train_trajectory_diffusion.py",
               "--csv-path", out_csv]
        if not args.dry_run:
            start = time.time()
            run_cmd(cmd, cwd=ROOT)
            log(f"Stage 2 completed in {time.time() - start:.1f}s", "SUCCESS")
            stages_completed.append("train_diffusion")
    
    if args.eval_diffusion:
        log("=== Stage 3: Diffusion Evaluation ===")
        cmd = [sys.executable, "analysis/safety_eval_diffusion.py"]
        if not args.dry_run:
            start = time.time()
            run_cmd(cmd, cwd=ROOT)
            log(f"Stage 3 completed in {time.time() - start:.1f}s", "SUCCESS")
            stages_completed.append("eval_diffusion")
    

    # Optional VLM annotation step
    if args.vlm_annotate:
        # Use csv-path if provided, else Stage 1 output
        pet_csv = Path(args.csv_path) if args.csv_path else Path(out_csv)
        if not pet_csv.exists():
            log(f"PET CSV {pet_csv} does not exist; skipping VLM annotation.", "WARN")
        else:
            vlm_csv = pet_csv.with_name(pet_csv.stem + "_vlm.csv")
            cmd = [
                sys.executable,
                "run_vlm_events.py",
                "--pet-csv", str(pet_csv),
                "--video", args.video,
                "--out-csv", str(vlm_csv),
            ]
            if not args.dry_run:
                start = time.time()
                run_cmd(cmd, cwd=ROOT)
                log(f"Stage 4 (VLM) completed in {time.time() - start:.1f}s", "SUCCESS")
                stages_completed.append("vlm")
            log(f"[INFO] VLM annotations written to {vlm_csv}")

    if args.export_video or args.live_bev_overlay:
        log("=== Stage 5/6: Video Overlay Export ===")
        start = time.time()
        overlay_csv = args.csv_path if args.csv_path else out_csv
        _export_overlay_videos(args.video, overlay_csv)
        log(f"Stage 5/6 completed in {time.time() - start:.1f}s", "SUCCESS")
        if args.export_video:
            stages_completed.append("export_video")
        if args.live_bev_overlay:
            stages_completed.append("live_bev_overlay")


    # Optional VLM annotation step
    if getattr(args, "vlm_annotate", False):
        if getattr(args, "csv_path", None):
            pet_csv = Path(args.csv_path)
        elif getattr(args, "out_csv", None):
            pet_csv = Path(args.out_csv)
        else:
            pet_csv = Path("outputs/petevents_working.csv")

        if not pet_csv.exists():
            print(f"[WARN] PET CSV {pet_csv} does not exist; skipping VLM annotation.")
        else:
            vlm_csv = pet_csv.with_name(pet_csv.stem + "_vlm.csv")
            cmd = [
                sys.executable,
                "run_vlm_events.py",
                "--pet-csv", str(pet_csv),
                "--video", args.video,
                "--out-csv", str(vlm_csv),
            ]
            if not args.dry_run:
                run_cmd(cmd, cwd=ROOT)
            print(f"[INFO] VLM annotations written to {vlm_csv}")

    effective_csv = args.csv_path if args.csv_path else out_csv

    if not args.dry_run:
        summary = {"pet_csv": effective_csv, "stages_completed": stages_completed,
                   "timestamp": datetime.now().isoformat()}
        summary_path = Path("outputs/research_run_summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2)
        log(f"Summary saved to {summary_path}")

    log(f"\n=== Done ===\nPET CSV: {effective_csv}")

if __name__ == "__main__":
    main()