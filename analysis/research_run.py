#!/usr/bin/env python3
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_cmd(cmd, cwd=ROOT):
    print("+", " ".join(shlex.quote(str(x)) for x in cmd))
    result = subprocess.run(cmd, cwd=cwd, text=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def default_csv_name(video_path: str, max_frames: int | None, pet_threshold: float) -> str:
    stem = Path(video_path).stem
    frame_tag = f"{max_frames}f" if max_frames is not None else "full"
    pet_tag = str(pet_threshold).replace(".", "p")
    return f"outputs/petevents_bev_{stem}_{frame_tag}_pet{pet_tag}.csv"


def main():
    parser = argparse.ArgumentParser(
        description="Research-friendly wrapper for NNDS video -> PET -> diffusion workflow"
    )
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--sam3-weights", default="sam3.pt", help="Path to SAM3 weights")
    parser.add_argument("--pet-threshold", type=float, default=2.0, help="PET threshold in seconds")
    parser.add_argument("--max-frames", type=int, default=None, help="Process only first N frames")
    parser.add_argument("--out-csv", default=None, help="Output PET CSV path")
    parser.add_argument("--train-diffusion", action="store_true", help="Run diffusion training after PET extraction")
    parser.add_argument("--eval-diffusion", action="store_true", help="Run diffusion evaluation after PET extraction")
    parser.add_argument("--train-csv-path", default=None, help="Optional CSV path for diffusion training")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use")

    args = parser.parse_args()

    out_csv = args.out_csv or default_csv_name(
        video_path=args.video,
        max_frames=args.max_frames,
        pet_threshold=args.pet_threshold,
    )

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)

    extract_cmd = [
        args.python,
        "traffic_analyzer.py",
        "--video", args.video,
        "--sam3-weights", args.sam3_weights,
        "--out-csv", out_csv,
        "--pet-threshold", str(args.pet_threshold),
    ]
    if args.max_frames is not None:
        extract_cmd += ["--max-frames", str(args.max_frames)]

    print("\n=== Stage 1: Video -> PET ===")
    print(f"Output CSV: {out_csv}")
    result = subprocess.run(extract_cmd, cwd=ROOT, env=env, text=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)

    train_csv = args.train_csv_path or out_csv

    if args.train_diffusion:
        print("\n=== Stage 2: Diffusion training ===")
        train_cmd = [
            args.python,
            "traffic_diffusion/train_trajectory_diffusion.py",
            "--csv-path", train_csv,
        ]
        result = subprocess.run(train_cmd, cwd=ROOT, env=env, text=True)
        if result.returncode != 0:
            raise SystemExit(result.returncode)

    if args.eval_diffusion:
        print("\n=== Stage 3: Diffusion evaluation ===")
        eval_cmd = [
            args.python,
            "analysis/safety_eval_diffusion.py",
        ]
        result = subprocess.run(eval_cmd, cwd=ROOT, env=env, text=True)
        if result.returncode != 0:
            raise SystemExit(result.returncode)

    print("\n=== Done ===")
    print(f"PET CSV: {out_csv}")
    if args.train_diffusion:
        print(f"Diffusion training CSV: {train_csv}")
    if args.eval_diffusion:
        print("Evaluation script completed.")


if __name__ == "__main__":
    main()
