#!/usr/bin/env python3
"""research_run.py - WORKING VERSION - Orchestrated NNDS research workflow"""


import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]

def log(msg, level="INFO"):
    print(f"[{level}] {msg}")

def run_cmd(cmd: List[str], cwd: Path = ROOT) -> None:
    log(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        log(f"Command failed with return code {result.returncode}", "ERROR")
        raise SystemExit(result.returncode)

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
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
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

    if not args.dry_run:
        summary = {"pet_csv": out_csv, "stages_completed": stages_completed,
                   "timestamp": datetime.now().isoformat()}
        summary_path = Path("outputs/research_run_summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2)
        log(f"Summary saved to {summary_path}")
    
    log(f"\n=== Done ===\nPET CSV: {out_csv}")


    # Optional VLM annotation step
    if getattr(args, "vlm_annotate", False):
        # Prefer explicit csv_path if given, else fall back to working PET CSV
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

if __name__ == "__main__":
    main()