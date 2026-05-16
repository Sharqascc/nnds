#!/usr/bin/env python3
"""research_run.py - Orchestrated NNDS research workflow."""

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
DEFAULT_OUTPUT_DIR = ROOT / "outputs"
OVERLAY_DIR = DEFAULT_OUTPUT_DIR / "video_overlays"


def log(msg: str, level: str = "INFO") -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")


def run_cmd(cmd: List[str], cwd: Path = ROOT, dry_run: bool = False) -> bool:
    """Run a command and return True if successful."""
    log(f"Executing: {' '.join(cmd)}")
    if dry_run:
        return True
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        log(f"Command failed with return code {result.returncode}", "ERROR")
        return False
    return True


def should_skip_stage(output_file: Path, force: bool = False) -> bool:
    """Check if a stage should be skipped because output already exists."""
    if force:
        return False
    return output_file.exists() and output_file.stat().st_size > 0


def _load_audit_logger():
    """Load audit logger if available; otherwise return None."""
    try:
        from analysis.audit_logger import get_audit_logger  # type: ignore
        return get_audit_logger()
    except Exception as e:
        log(f"Audit logging unavailable: {e}", "WARN")
        return None


def _audit_log(audit, event_id: str, action: str, metadata: Optional[dict] = None) -> None:
    """Safely write an audit log if logger is available."""
    if audit is None:
        return
    try:
        audit.log_action(
            event_id=event_id,
            action=action,
            reviewer_id="research_system",
            evidence_hash="",
            metadata=metadata or {},
        )
    except Exception as e:
        log(f"Audit logging failed for action {action}: {e}", "WARN")


def _parse_worldsample_traj(value):
    """Parse trajectory from serialized WorldSample / tuple-list formats."""
    if value is None:
        return []

    s = str(value).strip()
    if not s or s.lower() == "nan":
        return []

    parts = [p.strip() for p in s.split(";") if p.strip()]
    traj = []
    ws_pattern = re.compile(
        r"WorldSample\(t=([-+0-9.eE]+),\s*x=([-+0-9.eE]+),\s*y=([-+0-9.eE]+)\)"
    )

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
    """Convert world trajectories to a BEV canvas."""
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

    if abs(xmax - xmin) < 1e-6:
        xmin, xmax = xmin - 1.0, xmax + 1.0
    if abs(ymax - ymin) < 1e-6:
        ymin, ymax = ymin - 1.0, ymax + 1.0

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


def _export_overlay_videos(video_path: str, csv_path: str, fps: int = 30) -> bool:
    """Export overlay videos. Returns True on success."""
    csv_file = Path(csv_path)
    if not csv_file.exists():
        log(f"CSV not found: {csv_path}", "ERROR")
        return False

    df = pd.read_csv(csv_file)
    if df.empty:
        log("PET CSV is empty; skipping overlay video export.", "WARN")
        return False

    event_row = None
    for _, row in df.iterrows():
        traj_i = _parse_worldsample_traj(row.get("world_traj_i"))
        traj_j = _parse_worldsample_traj(row.get("world_traj_j"))
        if traj_i and traj_j:
            event_row = row
            break

    if event_row is None:
        log("No event with non-empty trajectories found; skipping overlay export.", "WARN")
        return False

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
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    analyzed_path = str(OVERLAY_DIR / "analyzed_video.mp4")
    bev_path = str(OVERLAY_DIR / "bev_overlay_video.mp4")

    try:
        plotter.generate_conflict_video(
            video_path=video_path,
            frame_range=(start_frame, end_frame),
            trajectories=[traj_i, traj_j],
            track_ids=track_ids,
            pet_value=pet_value,
            output_path=analyzed_path,
            fps=fps,
        )
        log(f"Analyzed video saved to {analyzed_path}")
    except Exception as e:
        log(f"Failed to generate analyzed video: {e}", "ERROR")
        return False

    def inset_callback(frame_idx):
        return _world_to_bev_canvas([traj_i, traj_j])

    try:
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
        log(f"BEV overlay video saved to {bev_path}")
    except Exception as e:
        log(f"Failed to generate BEV overlay video: {e}", "ERROR")
        return False

    return True


def generate_summary_report(effective_csv: str, stages: list, runtime: float, args) -> Path:
    """Generate a client-friendly summary report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "input_video": args.video,
        "effective_pet_csv": effective_csv,
        "stages_completed": stages,
        "total_runtime_seconds": runtime,
        "parameters": {
            "pet_threshold": args.pet_threshold,
            "max_frames": args.max_frames,
            "train_diffusion": args.train_diffusion,
            "eval_diffusion": args.eval_diffusion,
            "vlm_annotate": args.vlm_annotate,
            "export_video": args.export_video,
            "live_bev_overlay": args.live_bev_overlay,
            "force": args.force,
            "dry_run": args.dry_run,
        },
    }

    csv_file = Path(effective_csv)
    if csv_file.exists():
        try:
            df = pd.read_csv(csv_file)
            report["results"] = {
                "total_events": int(len(df)),
                "columns": list(df.columns),
                "has_pet_data": "pet" in df.columns,
            }
            if "pet" in df.columns and not df["pet"].dropna().empty:
                report["results"]["avg_pet"] = float(df["pet"].mean())
                report["results"]["min_pet"] = float(df["pet"].min())
                report["results"]["max_pet"] = float(df["pet"].max())
        except Exception as e:
            report["results_error"] = str(e)

    report_path = DEFAULT_OUTPUT_DIR / "research_run_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)

    return report_path


def main():
    parser = argparse.ArgumentParser(description="NNDS Research Pipeline")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--csv-path", help="Existing PET CSV to use for downstream analysis.")
    parser.add_argument("--vlm-annotate", action="store_true", help="Run VLM analysis on PET events.")
    parser.add_argument("--sam3-weights", default="sam3.pt", help="SAM3 weights")
    parser.add_argument("--pet-threshold", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--train-diffusion", action="store_true")
    parser.add_argument("--eval-diffusion", action="store_true")
    parser.add_argument("--export-video", action="store_true", help="Export analyzed conflict video.")
    parser.add_argument("--live-bev-overlay", action="store_true", help="Export video with BEV inset overlay.")
    parser.add_argument("--run-all", action="store_true", help="Run all major stages.")
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run stages even if outputs exist.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if args.run_all:
        args.train_diffusion = True
        args.eval_diffusion = True
        args.export_video = True
        args.live_bev_overlay = True

    audit = _load_audit_logger()
    run_id = f"RESEARCH_RUN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _audit_log(
        audit,
        run_id,
        "PIPELINE_STARTED",
        metadata={"args": {k: str(v) for k, v in vars(args).items()}},
    )

    video_path = Path(args.video)
    if not video_path.exists() and not args.skip_extraction:
        log(f"Video not found: {video_path}", "ERROR")
        _audit_log(audit, run_id, "PIPELINE_FAILED", metadata={"reason": f"Video not found: {video_path}"})
        sys.exit(1)

    if args.csv_path and not Path(args.csv_path).exists():
        log(f"CSV path does not exist: {args.csv_path}", "ERROR")
        _audit_log(audit, run_id, "PIPELINE_FAILED", metadata={"reason": f"CSV path not found: {args.csv_path}"})
        sys.exit(1)

    if not args.out_csv:
        stem = video_path.stem if args.video else "events"
        frame_tag = f"{args.max_frames}f" if args.max_frames else "full"
        pet_tag = str(args.pet_threshold).replace(".", "p")
        out_csv = str(DEFAULT_OUTPUT_DIR / f"petevents_bev_{stem}_{frame_tag}_pet{pet_tag}.csv")
    else:
        out_csv = args.out_csv

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stages_completed = []
    pipeline_start_time = time.time()

    effective_pet_csv = args.csv_path if args.csv_path else out_csv

    # Stage 1: Video -> PET
    if not args.skip_extraction and not args.csv_path:
        log(f"=== Stage 1: Video -> PET ({datetime.now().isoformat()}) ===")
        if should_skip_stage(out_path, args.force):
            log(f"Output exists, skipping extraction: {out_path}")
            stages_completed.append("extraction_skipped")
        else:
            cmd = [
                sys.executable,
                "traffic_analyzer.py",
                "--video",
                args.video,
                "--sam3-weights",
                args.sam3_weights,
                "--out-csv",
                out_csv,
                "--pet-threshold",
                str(args.pet_threshold),
            ]
            if args.max_frames:
                cmd += ["--max-frames", str(args.max_frames)]

            start = time.time()
            if run_cmd(cmd, cwd=ROOT, dry_run=args.dry_run):
                log(f"Stage 1 completed in {time.time() - start:.1f}s", "SUCCESS")
                stages_completed.append("extraction")
            else:
                log("Stage 1 failed", "ERROR")
                _audit_log(audit, run_id, "PIPELINE_FAILED", metadata={"stage": "extraction"})
                sys.exit(1)

    # Stage 2: Diffusion Training
    if args.train_diffusion:
        log(f"=== Stage 2: Diffusion Training ({datetime.now().isoformat()}) ===")
        start = time.time()
        cmd = [
            sys.executable,
            "traffic_diffusion/train_trajectory_diffusion.py",
            "--csv-path",
            effective_pet_csv,
        ]
        if run_cmd(cmd, cwd=ROOT, dry_run=args.dry_run):
            log(f"Stage 2 completed in {time.time() - start:.1f}s", "SUCCESS")
            stages_completed.append("train_diffusion")
        else:
            log("Stage 2 failed", "ERROR")

    # Stage 3: Diffusion Evaluation
    if args.eval_diffusion:
        log(f"=== Stage 3: Diffusion Evaluation ({datetime.now().isoformat()}) ===")
        start = time.time()
        cmd = [sys.executable, "analysis/safety_eval_diffusion.py"]
        if run_cmd(cmd, cwd=ROOT, dry_run=args.dry_run):
            log(f"Stage 3 completed in {time.time() - start:.1f}s", "SUCCESS")
            stages_completed.append("eval_diffusion")
        else:
            log("Stage 3 failed", "ERROR")

    # Stage 4: VLM Annotation
    if args.vlm_annotate:
        log(f"=== Stage 4: VLM Annotation ({datetime.now().isoformat()}) ===")
        pet_csv = Path(effective_pet_csv)
        if not pet_csv.exists() and not args.dry_run:
            log(f"PET CSV does not exist; skipping VLM annotation: {pet_csv}", "WARN")
        else:
            vlm_csv = pet_csv.with_name(pet_csv.stem + "_vlm.csv")
            start = time.time()
            cmd = [
                sys.executable,
                "run_vlm_events.py",
                "--pet-csv",
                str(pet_csv),
                "--video",
                args.video,
                "--out-csv",
                str(vlm_csv),
            ]
            if run_cmd(cmd, cwd=ROOT, dry_run=args.dry_run):
                log(f"Stage 4 completed in {time.time() - start:.1f}s", "SUCCESS")
                stages_completed.append("vlm")
                log(f"VLM annotations written to {vlm_csv}")
            else:
                log("Stage 4 failed", "ERROR")

    # Stage 5/6: Video export
    if args.export_video or args.live_bev_overlay:
        log(f"=== Stage 5/6: Video Overlay Export ({datetime.now().isoformat()}) ===")
        start = time.time()
        try:
            if args.dry_run:
                log("Dry run: skipping actual video overlay export.")
                ok = True
            else:
                ok = _export_overlay_videos(args.video, effective_pet_csv)

            if ok:
                log(f"Stage 5/6 completed in {time.time() - start:.1f}s", "SUCCESS")
                if args.export_video:
                    stages_completed.append("export_video")
                if args.live_bev_overlay:
                    stages_completed.append("live_bev_overlay")
            else:
                log("Stage 5/6 failed", "ERROR")
        except Exception as e:
            log(f"Video export failed: {e}", "ERROR")

    total_runtime = time.time() - pipeline_start_time

    if not args.dry_run:
        report_path = generate_summary_report(effective_pet_csv, stages_completed, total_runtime, args)
        log(f"Summary report saved to {report_path}", "SUCCESS")

        summary = {
            "pet_csv": effective_pet_csv,
            "stages_completed": stages_completed,
            "timestamp": datetime.now().isoformat(),
            "runtime_seconds": total_runtime,
        }
        summary_path = DEFAULT_OUTPUT_DIR / "research_run_summary.json"
        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2)
        log(f"Summary saved to {summary_path}")

    _audit_log(
        audit,
        run_id,
        "PIPELINE_COMPLETED",
        metadata={"stages": stages_completed, "runtime_seconds": total_runtime},
    )

    log(f"\n=== Done ===\nPET CSV: {effective_pet_csv}\nStages: {stages_completed}")


if __name__ == "__main__":
    main()
