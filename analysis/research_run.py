#!/usr/bin/env python3
"""
research_run.py

Orchestrated NNDS research workflow:
video -> PET extraction -> (optional) diffusion training -> (optional) diffusion evaluation.

Intended as a stable, research-friendly entry point for local runs and Colab.
"""

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]

logger = logging.getLogger(__name__)


class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def run_cmd(cmd: List[str], cwd: Path = ROOT, env: Optional[Dict[str, str]] = None) -> None:
    """Run a subprocess command, echoing it first, and exit on non-zero return."""
    logger.info("Executing: %s", " ".join(shlex.quote(str(x)) for x in cmd))
    result = subprocess.run(cmd, cwd=cwd, env=env, text=True)
    if result.returncode != 0:
        logger.error("Command failed with return code %d", result.returncode)
        raise SystemExit(result.returncode)


def default_csv_name(
    video_path: str, max_frames: Optional[int], pet_threshold: float
) -> str:
    """Construct a default PET CSV name based on video stem, frame regime, and PET threshold."""
    stem = Path(video_path).stem
    frame_tag = f"{max_frames}f" if max_frames is not None else "full"
    pet_tag = str(pet_threshold).replace(".", "p")
    return f"outputs/petevents_bev_{stem}_{frame_tag}_pet{pet_tag}.csv"


def save_summary(out_csv: str, stages_completed: List[str], output_path: Path) -> None:
    """Save a simple workflow summary to JSON."""
    summary = {
        "pet_csv": out_csv,
        "stages_completed": stages_completed,
        "timestamp": datetime.now().isoformat(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved workflow summary to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Research-friendly wrapper for NNDS video -> PET -> diffusion workflow"
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to input video",
    )
    parser.add_argument(
        "--sam3-weights",
        default="sam3.pt",
        help="Path to SAM3 weights",
    )
    parser.add_argument(
        "--pet-threshold",
        type=float,
        default=2.0,
        help="PET threshold in seconds",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Process only first N frames",
    )
    parser.add_argument(
        "--out-csv",
        default=None,
        help="Output PET CSV path (default: outputs/petevents_bev_<video>_<frames>_pet<th>.csv)",
    )
    parser.add_argument(
        "--train-diffusion",
        action="store_true",
        help="Run diffusion training after PET extraction",
    )
    parser.add_argument(
        "--eval-diffusion",
        action="store_true",
        help="Run diffusion evaluation after PET extraction",
    )
    parser.add_argument(
        "--train-csv-path",
        default=None,
        help="Optional CSV path for diffusion training (default: PET CSV from extraction stage)",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for subcommands",
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip Stage 1 (Video -> PET); assumes PET CSV already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional YAML config file to override CLI arguments",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("outputs/research_run_summary.json"),
        help="Path to JSON workflow summary",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing outputs without prompting",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def apply_config_overrides(args: argparse.Namespace) -> argparse.Namespace:
    """If a YAML config is provided, override matching attributes in args."""
    if not args.config:
        return args

    import yaml  # local import to keep baseline dependency light

    if not args.config.exists():
        raise SystemExit(f"Config file not found: {args.config}")

    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # Only override known attributes
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
            logger.info("Config override: %s = %r", key, value)
        else:
            logger.warning("Unknown config key (ignored): %s", key)

    return args


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    args = apply_config_overrides(args)

    stages_completed: List[str] = []

    # Basic validation
    video_path = Path(args.video)
    if not args.skip_extraction and not video_path.exists():
        logger.error("Input video not found: %s", video_path)
        raise SystemExit(1)

    analyzer_path = ROOT / "traffic_analyzer.py"
    if not analyzer_path.exists():
        logger.error("traffic_analyzer.py not found at %s", analyzer_path)
        raise SystemExit(1)

    out_csv = args.out_csv or default_csv_name(
        video_path=args.video,
        max_frames=args.max_frames,
        pet_threshold=args.pet_threshold,
    )

    out_csv_path = Path(out_csv)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)

    # Overwrite protection for PET CSV
    if (
        out_csv_path.exists()
        and not args.force
        and not args.skip_extraction
        and not args.dry_run
    ):
        response = input(f"{out_csv} exists. Overwrite? [y/N] ")
        if response.lower() != "y":
            print("Aborting.")
            return

    # Stage 1: Video -> PET
    if not args.skip_extraction:
        extract_cmd: List[str] = [
            args.python,
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
        if args.max_frames is not None:
            extract_cmd += ["--max-frames", str(args.max_frames)]

        logger.info("=== Stage 1: Video -> PET ===")
        logger.info("Output CSV: %s", out_csv)
        logger.debug("Command: %s", " ".join(shlex.quote(str(x)) for x in extract_cmd))

        start = time.time()
        if args.dry_run:
            print(Colors.YELLOW + "[DRY-RUN] Skipping execution of Stage 1." + Colors.RESET)
        else:
            run_cmd(extract_cmd, cwd=ROOT, env=env)
            elapsed = time.time() - start
            print(f"{Colors.GREEN}✓ Stage 1 completed in {elapsed:.1f} seconds{Colors.RESET}")
            stages_completed.append("extraction")
    else:
        logger.info("=== Stage 1: Skipped (Video -> PET) ===")
        logger.info("Assuming existing PET CSV at: %s", out_csv)
        if not out_csv_path.exists() and not args.dry_run:
            logger.warning(
                "WARNING: --skip-extraction set but PET CSV does not exist: %s", out_csv
            )

    # Train CSV (for diffusion) can be overridden
    train_csv = args.train_csv_path or out_csv

    # Stage 2: Diffusion training
    if args.train_diffusion:
        train_script = ROOT / "traffic_diffusion" / "train_trajectory_diffusion.py"
        if not train_script.exists():
            logger.error("Diffusion training script not found: %s", train_script)
            raise SystemExit(1)

        train_cmd: List[str] = [
            args.python,
            "traffic_diffusion/train_trajectory_diffusion.py",
            "--csv-path",
            train_csv,
        ]
        logger.info("=== Stage 2: Diffusion training ===")
        logger.info("Training CSV: %s", train_csv)
        logger.debug("Command: %s", " ".join(shlex.quote(str(x)) for x in train_cmd))

        start = time.time()
        if args.dry_run:
            print(Colors.YELLOW + "[DRY-RUN] Skipping execution of Stage 2." + Colors.RESET)
        else:
            run_cmd(train_cmd, cwd=ROOT, env=env)
            elapsed = time.time() - start
            print(f"{Colors.GREEN}✓ Stage 2 completed in {elapsed:.1f} seconds{Colors.RESET}")
            stages_completed.append("train_diffusion")

    # Stage 3: Diffusion evaluation
    if args.eval_diffusion:
        eval_script = ROOT / "analysis" / "safety_eval_diffusion.py"
        if not eval_script.exists():
            logger.error("Diffusion evaluation script not found: %s", eval_script)
            raise SystemExit(1)

        eval_cmd: List[str] = [
            args.python,
            "analysis/safety_eval_diffusion.py",
        ]
        logger.info("=== Stage 3: Diffusion evaluation ===")
        logger.debug("Command: %s", " ".join(shlex.quote(str(x)) for x in eval_cmd))

        start = time.time()
        if args.dry_run:
            print(Colors.YELLOW + "[DRY-RUN] Skipping execution of Stage 3." + Colors.RESET)
        else:
            run_cmd(eval_cmd, cwd=ROOT, env=env)
            elapsed = time.time() - start
            print(f"{Colors.GREEN}✓ Stage 3 completed in {elapsed:.1f} seconds{Colors.RESET}")
            stages_completed.append("eval_diffusion")

    # Save a simple JSON summary
    if not args.dry_run:
        save_summary(out_csv, stages_completed, args.summary_json)

    print("\n=== Done ===")
    print(f"PET CSV: {out_csv}")
    if args.train_diffusion:
        print(f"Diffusion training CSV: {train_csv}")
    if args.eval_diffusion:
        print("Evaluation script completed.")


if __name__ == "__main__":
    main()
