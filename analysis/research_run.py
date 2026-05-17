#!/usr/bin/env python
"""
NNDS Research Workflow Orchestrator

End-to-end orchestration for:
1. PET extraction from traffic video
2. PET statistical summary and exports
3. Optional gate counting
4. Optional diffusion model training
5. Optional diffusion safety evaluation

Designed to match the NNDS repository structure documented in README.md.

Example usage:
    PYTHONPATH=. python analysis/research_run.py \
        --video videos/traffic_video.mp4 \
        --train-diffusion --eval-diffusion

    PYTHONPATH=. python analysis/research_run.py \
        --video videos/traffic_video.mp4 \
        --skip-extraction --train-diffusion

    PYTHONPATH=. python analysis/research_run.py \
        --video videos/traffic_video.mp4 \
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

VERSION = "1.0.0"


@dataclass
class WorkflowPaths:
    repo_root: Path
    video: Path
    bev_config: Path
    grid_config: Path
    gate_config: Optional[Path]
    sam3_weights: Path
    rtdetr_weights: Path
    output_dir: Path
    pet_csv: Path
    summary_dir: Path
    gate_dir: Path
    diffusion_dir: Path
    logs_dir: Path


@dataclass
class WorkflowResult:
    extraction_ran: bool = False
    summary_ran: bool = False
    gate_count_ran: bool = False
    diffusion_train_ran: bool = False
    diffusion_eval_ran: bool = False
    pet_csv: Optional[str] = None
    summary_exports: Optional[Dict[str, str]] = None
    gate_csv: Optional[str] = None
    diffusion_outputs: Optional[Dict[str, str]] = None
    notes: Optional[List[str]] = None


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    else:
        logging.getLogger().setLevel(level)


def infer_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_paths(args: argparse.Namespace) -> WorkflowPaths:
    repo_root = infer_repo_root()

    output_dir = (repo_root / args.output_dir).resolve()
    pet_csv = (repo_root / args.pet_csv).resolve()
    summary_dir = (repo_root / args.summary_dir).resolve()
    gate_dir = (repo_root / args.gate_dir).resolve()
    diffusion_dir = (repo_root / args.diffusion_dir).resolve()
    logs_dir = (repo_root / args.logs_dir).resolve()

    gate_config = (repo_root / args.gate_config).resolve() if args.gate_config else None

    return WorkflowPaths(
        repo_root=repo_root,
        video=(repo_root / args.video).resolve(),
        bev_config=(repo_root / args.bev_config).resolve(),
        grid_config=(repo_root / args.grid_config).resolve(),
        gate_config=gate_config,
        sam3_weights=(repo_root / args.sam3_weights).resolve(),
        rtdetr_weights=(repo_root / args.rtdetr_weights).resolve(),
        output_dir=output_dir,
        pet_csv=pet_csv,
        summary_dir=summary_dir,
        gate_dir=gate_dir,
        diffusion_dir=diffusion_dir,
        logs_dir=logs_dir,
    )


def ensure_dirs(paths: WorkflowPaths) -> None:
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    paths.summary_dir.mkdir(parents=True, exist_ok=True)
    paths.gate_dir.mkdir(parents=True, exist_ok=True)
    paths.diffusion_dir.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    paths.pet_csv.parent.mkdir(parents=True, exist_ok=True)


def validate_required_inputs(
    paths: WorkflowPaths,
    detector: str,
    skip_extraction: bool,
    allow_missing_weights_in_skip_mode: bool = True,
) -> None:
    if not paths.video.exists():
        raise FileNotFoundError(f"Video not found: {paths.video}")

    if not skip_extraction:
        if not paths.bev_config.exists():
            raise FileNotFoundError(f"BEV config not found: {paths.bev_config}")
        if not paths.grid_config.exists():
            raise FileNotFoundError(f"Grid config not found: {paths.grid_config}")

    if detector == "sam3":
        if (not paths.sam3_weights.exists()) and not (skip_extraction and allow_missing_weights_in_skip_mode):
            raise FileNotFoundError(
                f"SAM3 weights not found: {paths.sam3_weights}\n"
                "Download from Hugging Face, e.g.:\n"
                "https://huggingface.co/sharqascc/sam3-traffic-model/resolve/main/sam3.pt"
            )
    elif detector == "rtdetr":
        if (not paths.rtdetr_weights.exists()) and not (skip_extraction and allow_missing_weights_in_skip_mode):
            raise FileNotFoundError(f"RT-DETR weights not found: {paths.rtdetr_weights}")
    else:
        raise ValueError(f"Unsupported detector: {detector}")

    if skip_extraction and not paths.pet_csv.exists():
        raise FileNotFoundError(
            f"--skip-extraction was set, but PET CSV does not exist: {paths.pet_csv}"
        )


def dry_run_report(args: argparse.Namespace, paths: WorkflowPaths) -> None:
    plan = {
        "repo_root": str(paths.repo_root),
        "video": str(paths.video),
        "bev_config": str(paths.bev_config),
        "grid_config": str(paths.grid_config),
        "gate_config": str(paths.gate_config) if paths.gate_config else None,
        "sam3_weights": str(paths.sam3_weights),
        "rtdetr_weights": str(paths.rtdetr_weights),
        "pet_csv": str(paths.pet_csv),
        "summary_dir": str(paths.summary_dir),
        "gate_dir": str(paths.gate_dir),
        "diffusion_dir": str(paths.diffusion_dir),
        "stages": {
            "skip_extraction": args.skip_extraction,
            "run_summary": not args.skip_summary,
            "run_gate_count": args.run_gate_count,
            "train_diffusion": args.train_diffusion,
            "eval_diffusion": args.eval_diffusion,
        },
        "detector": args.detector,
        "max_frames": args.max_frames,
        "pet_threshold": args.pet_threshold,
    }
    print(json.dumps(plan, indent=2))


def run_pet_extraction(args: argparse.Namespace, paths: WorkflowPaths) -> pd.DataFrame:
    logger.info("Running PET extraction...")
    from traffic_analyzer import run_video_to_pet

    df = run_video_to_pet(
        video_path=paths.video,
        bev_config_path=paths.bev_config,
        grid_config_path=paths.grid_config,
        sam3_weights_path=paths.sam3_weights,
        out_csv_path=paths.pet_csv,
        pet_threshold=args.pet_threshold,
        max_frames=args.max_frames,
        show_progress=not args.no_progress,
        detector=args.detector,
        rtdetr_weights_path=paths.rtdetr_weights,
    )
    logger.info("PET extraction completed: %s rows -> %s", len(df), paths.pet_csv)
    return df


def run_pet_summary(args: argparse.Namespace, paths: WorkflowPaths) -> Dict[str, str]:
    logger.info("Running PET summary...")
    from analysis.pet_summary import PETEventAnalyzer

    thresholds = {
        "critical": args.critical,
        "serious": args.serious if args.serious is not None else (args.critical + args.moderate) / 2.0,
        "moderate": args.moderate,
        "safe": args.moderate,
    }

    analyzer = PETEventAnalyzer(
        csv_path=paths.pet_csv,
        conflict_col=args.conflict_col,
        thresholds=thresholds,
    )

    analyzer.print_summary(show_risk_buckets=not args.no_risk_buckets)

    exported = analyzer.export_results(
        output_dir=paths.summary_dir,
        baseline_csv=Path(args.baseline_csv).resolve() if args.baseline_csv else None,
        fmt=args.summary_format,
    )

    exported_str = {k: str(v) for k, v in exported.items()}
    logger.info("PET summary exports: %s", exported_str)
    return exported_str


def run_gate_count(args: argparse.Namespace, paths: WorkflowPaths) -> Optional[str]:
    logger.info("Running gate counting...")
    if paths.gate_config is None:
        logger.warning("Gate counting requested but no gate config provided.")
        return None
    if not paths.gate_config.exists():
        logger.warning("Gate config not found: %s", paths.gate_config)
        return None

    from gate_counter import TrafficVolumeCounter

    out_video = str(paths.gate_dir / "gate_count_overlay.mp4")
    out_csv = paths.gate_dir / "gate_counts.csv"

    counter = TrafficVolumeCounter(
        video_path=str(paths.video),
        gate_config=str(paths.gate_config),
        min_confidence=args.gate_min_confidence,
        draw_stats=True,
        draw_tracks=args.draw_gate_tracks,
    )

    detector_name = args.gate_detector_module
    if detector_name:
        module_name, func_name = detector_name.rsplit(":", 1)
        mod = __import__(module_name, fromlist=[func_name])
        detector_fn = getattr(mod, func_name)
    else:
        raise ValueError(
            "Gate counting requires --gate-detector-module in the format module.submodule:function"
        )

    result = counter.process_video(
        detector=detector_fn,
        output_video=out_video,
        max_frames=args.max_frames,
        log_visual_debug=args.gate_visual_debug,
        show_progress=not args.no_progress,
    )
    counter.save_results(result, out_csv)
    logger.info("Gate counting completed -> %s", out_csv)
    return str(out_csv)


def run_diffusion_training(args: argparse.Namespace, paths: WorkflowPaths) -> Dict[str, str]:
    logger.info("Running diffusion training...")
    outputs: Dict[str, str] = {}

    try:
        from traffic_diffusion.train_trajectory_diffusion import main as train_main
    except Exception as exc:
        raise RuntimeError(
            "Could not import traffic_diffusion.train_trajectory_diffusion.main"
        ) from exc

    train_argv = [
        "--csv-path", str(paths.pet_csv),
        "--epochs", str(args.diffusion_epochs),
    ]
    if args.diffusion_batch_size is not None:
        train_argv += ["--batch-size", str(args.diffusion_batch_size)]
    if args.diffusion_lr is not None:
        train_argv += ["--lr", str(args.diffusion_lr)]
    if args.diffusion_output_dir is not None:
        train_argv += ["--output-dir", str((paths.repo_root / args.diffusion_output_dir).resolve())]
    else:
        train_argv += ["--output-dir", str(paths.diffusion_dir)]

    train_main(train_argv)
    outputs["train_output_dir"] = str(
        (paths.repo_root / args.diffusion_output_dir).resolve()
        if args.diffusion_output_dir is not None
        else paths.diffusion_dir
    )
    logger.info("Diffusion training finished.")
    return outputs


def run_diffusion_eval(args: argparse.Namespace, paths: WorkflowPaths) -> Dict[str, str]:
    logger.info("Running diffusion safety evaluation...")
    outputs: Dict[str, str] = {}

    summary_csv = paths.output_dir / "safety_eval_diffusion_summary.csv"
    events_csv = paths.output_dir / "safety_events_diffusion_model.csv"

    try:
        from analysis.safety_eval_diffusion import main as eval_main
    except Exception as exc:
        raise RuntimeError(
            "Could not import analysis.safety_eval_diffusion.main"
        ) from exc

    eval_argv: List[str] = []
    if args.eval_checkpoint:
        eval_argv += ["--checkpoint", str((paths.repo_root / args.eval_checkpoint).resolve())]
    if args.eval_num_samples is not None:
        eval_argv += ["--num-samples", str(args.eval_num_samples)]
    if args.eval_output_dir is not None:
        eval_argv += ["--output-dir", str((paths.repo_root / args.eval_output_dir).resolve())]
    else:
        eval_argv += ["--output-dir", str(paths.output_dir)]
    if args.pet_csv_for_eval:
        eval_argv += ["--csv-path", str((paths.repo_root / args.pet_csv_for_eval).resolve())]
    else:
        eval_argv += ["--csv-path", str(paths.pet_csv)]

    eval_main(eval_argv)

    outputs["safety_events_csv"] = str(events_csv)
    outputs["safety_summary_csv"] = str(summary_csv)

    if args.compare_diffusion_pet:
        try:
            from analysis.pet_diffusion_analysis import PETDiffusionAnalyzer
            analyzer = PETDiffusionAnalyzer()
            logger.info("PETDiffusionAnalyzer imported successfully for downstream comparison.")
            outputs["pet_diffusion_analysis"] = "available"
        except Exception:
            logger.warning("PET diffusion analysis module exists but could not be initialized.")

    logger.info("Diffusion safety evaluation finished.")
    return outputs


def save_manifest(paths: WorkflowPaths, args: argparse.Namespace, result: WorkflowResult) -> Path:
    manifest_path = paths.logs_dir / "research_run_manifest.json"
    payload = {
        "version": VERSION,
        "paths": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(paths).items()},
        "args": vars(args),
        "result": asdict(result),
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved manifest -> %s", manifest_path)
    return manifest_path


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NNDS research workflow orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--video", required=True, help="Input video path, relative to repo root")
    parser.add_argument("--bev-config", default="configs/bev_config.json", help="BEV config path")
    parser.add_argument("--grid-config", default="configs/GITI_grid_config.json", help="Grid config path")
    parser.add_argument("--gate-config", default="configs/gate_config.yaml", help="Gate config path")
    parser.add_argument("--sam3-weights", default="sam3.pt", help="SAM3 weights path")
    parser.add_argument("--rtdetr-weights", default="rtdetr-l.pt", help="RT-DETR weights path")
    parser.add_argument("--detector", choices=["sam3", "rtdetr"], default="sam3", help="Detection backend")

    parser.add_argument("--output-dir", default="outputs", help="Main output directory")
    parser.add_argument("--pet-csv", default="outputs/petevents_bev.csv", help="PET CSV output path")
    parser.add_argument("--summary-dir", default="analysis_results", help="PET summary export directory")
    parser.add_argument("--gate-dir", default="outputs/gates", help="Gate counting output directory")
    parser.add_argument("--diffusion-dir", default="outputs/diffusion", help="Diffusion output directory")
    parser.add_argument("--logs-dir", default="outputs/logs", help="Logs and manifest output directory")

    parser.add_argument("--pet-threshold", type=float, default=2.0, help="PET threshold in seconds")
    parser.add_argument("--max-frames", type=int, default=None, help="Process only first N frames")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars")

    parser.add_argument("--skip-extraction", action="store_true", help="Skip PET extraction and reuse existing PET CSV")
    parser.add_argument("--skip-summary", action="store_true", help="Skip PET summary stage")
    parser.add_argument("--run-gate-count", action="store_true", help="Run gate counting stage")
    parser.add_argument("--train-diffusion", action="store_true", help="Run diffusion training stage")
    parser.add_argument("--eval-diffusion", action="store_true", help="Run diffusion evaluation stage")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved workflow plan without executing")

    parser.add_argument("--conflict-col", default="conflict_type", help="Conflict type column for PET analysis")
    parser.add_argument("--baseline-csv", default=None, help="Optional baseline PET CSV for comparison")
    parser.add_argument("--critical", type=float, default=1.0, help="Critical PET threshold")
    parser.add_argument("--serious", type=float, default=None, help="Serious PET threshold; defaults to midpoint")
    parser.add_argument("--moderate", type=float, default=3.0, help="Moderate PET threshold")
    parser.add_argument("--summary-format", choices=["json", "csv"], default="json", help="PET summary export format")
    parser.add_argument("--no-risk-buckets", action="store_true", help="Hide risk bucket console output")

    parser.add_argument(
        "--gate-detector-module",
        default=None,
        help="Detector function for gate counting, format: module.submodule:function",
    )
    parser.add_argument("--gate-min-confidence", type=float, default=0.25, help="Gate counter detection confidence")
    parser.add_argument("--draw-gate-tracks", action="store_true", help="Render gate tracking overlays")
    parser.add_argument("--gate-visual-debug", action="store_true", help="Verbose gate visual debug")

    parser.add_argument("--diffusion-epochs", type=int, default=100, help="Diffusion training epochs")
    parser.add_argument("--diffusion-batch-size", type=int, default=None, help="Optional diffusion batch size")
    parser.add_argument("--diffusion-lr", type=float, default=None, help="Optional diffusion learning rate")
    parser.add_argument("--diffusion-output-dir", default=None, help="Optional custom diffusion output dir")

    parser.add_argument("--eval-checkpoint", default=None, help="Checkpoint path for diffusion evaluation")
    parser.add_argument("--eval-num-samples", type=int, default=None, help="Number of sampled futures for evaluation")
    parser.add_argument("--eval-output-dir", default=None, help="Optional custom evaluation output dir")
    parser.add_argument("--pet-csv-for-eval", default=None, help="Optional PET CSV to use for evaluation stage")
    parser.add_argument("--compare-diffusion-pet", action="store_true", help="Try loading PET diffusion comparison tools")

    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    setup_logging(args.verbose)

    paths = build_paths(args)
    ensure_dirs(paths)

    result = WorkflowResult(notes=[])

    try:
        validate_required_inputs(
            paths=paths,
            detector=args.detector,
            skip_extraction=args.skip_extraction,
        )

        if args.dry_run:
            dry_run_report(args, paths)
            return

        if not args.skip_extraction:
            run_pet_extraction(args, paths)
            result.extraction_ran = True
        else:
            logger.info("Skipping PET extraction; using existing CSV: %s", paths.pet_csv)
            result.notes.append("Extraction skipped; existing PET CSV reused.")

        result.pet_csv = str(paths.pet_csv)

        if not args.skip_summary:
            exports = run_pet_summary(args, paths)
            result.summary_ran = True
            result.summary_exports = exports
        else:
            logger.info("Skipping PET summary stage.")
            result.notes.append("PET summary skipped.")

        if args.run_gate_count:
            gate_csv = run_gate_count(args, paths)
            if gate_csv is not None:
                result.gate_count_ran = True
                result.gate_csv = gate_csv
            else:
                result.notes.append("Gate count requested but skipped due to missing config or detector.")
        else:
            logger.info("Gate count stage not requested.")

        if args.train_diffusion:
            diffusion_train_outputs = run_diffusion_training(args, paths)
            result.diffusion_train_ran = True
            result.diffusion_outputs = result.diffusion_outputs or {}
            result.diffusion_outputs.update(diffusion_train_outputs)
        else:
            logger.info("Diffusion training stage not requested.")

        if args.eval_diffusion:
            diffusion_eval_outputs = run_diffusion_eval(args, paths)
            result.diffusion_eval_ran = True
            result.diffusion_outputs = result.diffusion_outputs or {}
            result.diffusion_outputs.update(diffusion_eval_outputs)
        else:
            logger.info("Diffusion evaluation stage not requested.")

        save_manifest(paths, args, result)

        print(json.dumps(asdict(result), indent=2))

    except Exception as exc:
        logger.error("Research workflow failed: %s", exc)
        logger.debug(traceback.format_exc())
        fail_result = asdict(result)
        fail_result["error"] = str(exc)
        print(json.dumps(fail_result, indent=2), file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
