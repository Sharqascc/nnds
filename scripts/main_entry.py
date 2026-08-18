from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

def find_repo_root(start: Optional[Path] = None) -> Path:
    start = start or Path.cwd()
    for p in [start, *start.parents]:
        if (p / ".git").exists() and (p / "analysis").exists():
            return p
    return start

ROOT = find_repo_root()
DEFAULT_RESEARCH_RUN = ROOT / "analysis" / "research_run.py"
DEFAULT_EXPORTER = ROOT / "master_outputs_exporter.py"
DEFAULT_SAMPLE_VIDEO = ROOT / "sample_data" / "traffic_video.mp4"

@dataclass
class StepResult:
    name: str
    cmd: List[str]
    returncode: int

def prompt_yes_no(question: str, default: bool = False) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    ans = input(f"{question} {suffix} ").strip().lower()
    if not ans:
        return default
    return ans in {"y", "yes"}

def run_step(name: str, cmd: List[str], cwd: Path = ROOT, dry_run: bool = False) -> StepResult:
    print(f"\n=== {name} ===")
    print("CMD:", " ".join(cmd))
    if dry_run:
        return StepResult(name=name, cmd=cmd, returncode=0)
    proc = subprocess.run(cmd, cwd=str(cwd))
    return StepResult(name=name, cmd=cmd, returncode=proc.returncode)

def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive NNDS main entry point")
    p.add_argument("--video", default=None)
    p.add_argument("--sam3-weights", default="sam3.pt")
    p.add_argument("--out-csv", default=None)
    p.add_argument("--pet-threshold", type=float, default=2.0)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--detector", default="uvh-coco-fused")
    p.add_argument("--uvh-model", default=None)
    p.add_argument("--coco-person-model", default=None)
    p.add_argument("--uvh-conf", type=float, default=0.20)
    p.add_argument("--coco-person-conf", type=float, default=0.20)
    p.add_argument("--detector-imgsz", type=int, default=1280)
    p.add_argument("--person-suppress-overlap", type=float, default=0.35)
    p.add_argument("--auto-approve", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-diffusion-train", action="store_true")
    p.add_argument("--skip-diffusion-eval", action="store_true")
    p.add_argument("--skip-export", action="store_true")
    args, _ = p.parse_known_args()
    return args

def resolve_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if not args.video:
        args.video = str(DEFAULT_SAMPLE_VIDEO) if DEFAULT_SAMPLE_VIDEO.exists() else input("Enter video path: ").strip()
    if args.uvh_model is None:
        args.uvh_model = str(Path.home() / ".cache" / "huggingface" / "hub" / "models--iisc-aiml" / "UVH-26" / "snapshots" / "4a22412775adb6f97f22735647afee976b4638a0" / "weights" / "YOLOv11-SUVH-26-MV-YOLOv11-S.pt")
    if args.coco_person_model is None:
        args.coco_person_model = "yolo26m-seg.pt"
    if args.out_csv is None:
        stem = Path(args.video).stem
        pet_tag = str(args.pet_threshold).replace('.', 'p')
        args.out_csv = str(ROOT / "outputs" / f"petevents_bev_{stem}_pet{pet_tag}.csv")
    return args

def load_csv_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")

def main() -> int:
    args = resolve_defaults(build_args())
    if not Path(args.video).exists():
        raise FileNotFoundError(f"Video not found: {args.video}")

    plan = [
        ("run_pet_pipeline", [
            sys.executable, str(DEFAULT_RESEARCH_RUN),
            "--video", args.video,
            "--sam3-weights", args.sam3_weights,
            "--out-csv", args.out_csv,
            "--pet-threshold", str(args.pet_threshold),
            "--detector", args.detector,
            "--uvh-model", args.uvh_model,
            "--coco-person-model", args.coco_person_model,
            "--uvh-conf", str(args.uvh_conf),
            "--coco-person-conf", str(args.coco_person_conf),
            "--detector-imgsz", str(args.detector_imgsz),
            "--person-suppress-overlap", str(args.person_suppress_overlap),
        ] + (["--max-frames", str(args.max_frames)] if args.max_frames is not None else []))
    ]

    if not args.skip_diffusion_train:
        plan.append(("train_diffusion", [sys.executable, str(ROOT / "analysis" / "trafficdiffusiontraintrajectorydiffusion.py"), "--csv-path", args.out_csv]))
    if not args.skip_diffusion_eval:
        plan.append(("eval_diffusion", [sys.executable, str(ROOT / "analysis" / "analysissafetyevaldiffusion.py")]))

    print(f"Repo root: {ROOT}")
    print(f"Video: {args.video}")
    print("\nPlanned steps:")
    for i, (name, cmd) in enumerate(plan, start=1):
        print(f"{i}. {name}: {' '.join(cmd)}")

    if not args.auto_approve and not prompt_yes_no("Go ahead with the planned steps?", default=False):
        print("Aborted.")
        return 1

    for name, cmd in plan:
        if not args.auto_approve and not prompt_yes_no(f"Run step '{name}'?", default=True):
            print(f"Skipped {name}")
            continue
        res = run_step(name, cmd, cwd=ROOT, dry_run=args.dry_run)
        if res.returncode != 0:
            print(f"Step failed: {name}")
            return res.returncode

    if not args.skip_export and DEFAULT_EXPORTER.exists():
        try:
            from master_outputs_exporter import export_master_outputs
            conflicts = load_csv_records(Path(args.out_csv))
            tracks_csv = Path(args.out_csv).with_name(Path(args.out_csv).stem + "_detections.csv")
            tracks = load_csv_records(tracks_csv)
            run_meta = {
                "run_id": Path(args.out_csv).stem,
                "source_video": args.video,
                "frames_processed": args.max_frames,
                "fps": None,
                "detector": args.detector,
                "uvh_model": args.uvh_model,
                "coco_person_model": args.coco_person_model,
                "uvh_conf": args.uvh_conf,
                "coco_person_conf": args.coco_person_conf,
                "detector_imgsz": args.detector_imgsz,
                "person_suppress_overlap": args.person_suppress_overlap,
                "pet_threshold_s": args.pet_threshold,
                "version": "1.0",
            }
            result = export_master_outputs(conflicts, tracks, run_meta, out_dir=str(ROOT / "outputs"))
            print("Master export written:")
            print(result)
        except Exception as e:
            print(f"Master export failed: {e}")

    print("\nDone.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
