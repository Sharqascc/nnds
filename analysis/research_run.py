#!/usr/bin/env python
"""
NNDS Research Workflow Orchestrator

Integrated stages:
1. PET extraction
2. PET summary
3. Optional gate counting
4. Optional diffusion training
5. Optional diffusion evaluation
6. Optional PET-threshold sensitivity
7. Optional detector comparison
8. Optional ground-truth comparison metrics
9. Optional significance testing
10. Optional diffusion cross-validation
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)
VERSION = "1.2.0"


@dataclass
class WorkflowPaths:
    repo_root: Path
    video: Optional[Path]
    videos_list: Optional[Path]
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
    validation_dir: Path
    logs_dir: Path


@dataclass
class WorkflowResult:
    extraction_ran: bool = False
    summary_ran: bool = False
    gate_count_ran: bool = False
    diffusion_train_ran: bool = False
    diffusion_eval_ran: bool = False
    cross_validation_ran: bool = False
    ground_truth_eval_ran: bool = False
    significance_tests_ran: bool = False
    pet_csv: Optional[str] = None
    summary_exports: Optional[Dict[str, str]] = None
    gate_csv: Optional[str] = None
    diffusion_outputs: Optional[Dict[str, str]] = None
    validation_outputs: Optional[Dict[str, Any]] = None
    warnings: Optional[List[str]] = None
    notes: Optional[List[str]] = None
    qa: Optional[Dict[str, Any]] = None


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


def infer_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_paths(args: argparse.Namespace) -> WorkflowPaths:
    repo_root = infer_repo_root()
    gate_config = (repo_root / args.gate_config).resolve() if args.gate_config else None
    return WorkflowPaths(
        repo_root=repo_root,
        video=(repo_root / args.video).resolve() if args.video else None,
        videos_list=(repo_root / args.videos_list).resolve() if args.videos_list else None,
        bev_config=(repo_root / args.bev_config).resolve(),
        grid_config=(repo_root / args.grid_config).resolve(),
        gate_config=gate_config,
        sam3_weights=(repo_root / args.sam3_weights).resolve(),
        rtdetr_weights=(repo_root / args.rtdetr_weights).resolve(),
        output_dir=(repo_root / args.output_dir).resolve(),
        pet_csv=(repo_root / args.pet_csv).resolve(),
        summary_dir=(repo_root / args.summary_dir).resolve(),
        gate_dir=(repo_root / args.gate_dir).resolve(),
        diffusion_dir=(repo_root / args.diffusion_dir).resolve(),
        validation_dir=(repo_root / args.validation_dir).resolve(),
        logs_dir=(repo_root / args.logs_dir).resolve(),
    )


def ensure_dirs(paths: WorkflowPaths) -> None:
    for p in [
        paths.output_dir,
        paths.summary_dir,
        paths.gate_dir,
        paths.diffusion_dir,
        paths.validation_dir,
        paths.logs_dir,
        paths.pet_csv.parent,
    ]:
        p.mkdir(parents=True, exist_ok=True)


def resolve_thresholds(args: argparse.Namespace) -> Dict[str, float]:
    serious = args.serious if args.serious is not None else (args.critical + args.moderate) / 2.0
    if not (args.critical <= serious <= args.moderate):
        raise ValueError(
            f"Invalid thresholds: expected critical <= serious <= moderate, got "
            f"{args.critical}, {serious}, {args.moderate}"
        )
    return {
        "critical": float(args.critical),
        "serious": float(serious),
        "moderate": float(args.moderate),
        "safe": float(args.moderate),
    }


def validate_required_inputs(paths: WorkflowPaths, detector: str, skip_extraction: bool) -> None:
    if not paths.video and not paths.videos_list:
        raise ValueError("Provide either --video or --videos-list")

    if paths.video and not paths.video.exists():
        raise FileNotFoundError(f"Video not found: {paths.video}")
    if paths.videos_list and not paths.videos_list.exists():
        raise FileNotFoundError(f"Videos list not found: {paths.videos_list}")

    if not skip_extraction:
        if not paths.bev_config.exists():
            raise FileNotFoundError(f"BEV config not found: {paths.bev_config}")
        if not paths.grid_config.exists():
            raise FileNotFoundError(f"Grid config not found: {paths.grid_config}")

    if detector == "sam3" and not skip_extraction and not paths.sam3_weights.exists():
        raise FileNotFoundError(
            f"SAM3 weights not found: {paths.sam3_weights}\n"
            "Download from Hugging Face:\n"
            "https://huggingface.co/sharqascc/sam3-traffic-model/resolve/main/sam3.pt"
        )
    if detector == "rtdetr" and not skip_extraction and not paths.rtdetr_weights.exists():
        raise FileNotFoundError(f"RT-DETR weights not found: {paths.rtdetr_weights}")

    if skip_extraction and not paths.pet_csv.exists():
        raise FileNotFoundError(f"--skip-extraction set, but PET CSV does not exist: {paths.pet_csv}")


def load_video_list(videos_list: Path) -> List[str]:
    lines = [x.strip() for x in videos_list.read_text(encoding="utf-8").splitlines()]
    return [x for x in lines if x and not x.startswith("#")]


def run_pet_extraction(args: argparse.Namespace, paths: WorkflowPaths, video_path: Path, out_csv: Path) -> pd.DataFrame:
    logger.info("Running PET extraction on %s", video_path)
    from traffic_analyzer import run_video_to_pet

    kwargs = dict(
        video_path=video_path,
        bev_config_path=paths.bev_config,
        grid_config_path=paths.grid_config,
        sam3_weights_path=paths.sam3_weights,
        out_csv_path=out_csv,
        pet_threshold=args.pet_threshold,
        max_frames=args.max_frames,
        show_progress=not args.no_progress,
        detector=args.detector,
        rtdetr_weights_path=paths.rtdetr_weights,
    )
    if args.imgsz is not None:
        kwargs["imgsz"] = args.imgsz

    df = run_video_to_pet(**kwargs)
    logger.info("PET extraction completed: %s rows -> %s", len(df), out_csv)
    return df


def run_pet_summary(
    args: argparse.Namespace,
    pet_csv: Path,
    summary_dir: Path,
    thresholds: Dict[str, float],
) -> Dict[str, str]:
    logger.info(
        "Running PET summary with thresholds: critical<%.3f, serious<%.3f, moderate<%.3f, safe>=%.3f",
        thresholds["critical"], thresholds["serious"], thresholds["moderate"], thresholds["safe"]
    )
    from analysis.pet_summary import PETEventAnalyzer

    analyzer = PETEventAnalyzer(
        csv_path=pet_csv,
        conflict_col=args.conflict_col,
        thresholds=thresholds,
    )
    analyzer.print_summary(show_risk_buckets=not args.no_risk_buckets)

    exported = analyzer.export_results(
        output_dir=summary_dir,
        baseline_csv=Path(args.baseline_csv).resolve() if args.baseline_csv else None,
        fmt=args.summary_format,
    )
    exported_str = {k: str(v) for k, v in exported.items()}
    logger.info("PET summary exports: %s", exported_str)
    return exported_str


def evaluate_pet_quality(pet_csv: Path, max_frames: Optional[int]) -> Dict[str, Any]:
    df = pd.read_csv(pet_csv)
    qa: Dict[str, Any] = {"n_events": int(len(df)), "warnings": []}

    if len(df) == 0:
        qa["warnings"].append("No PET events extracted.")
        return qa

    if "pet" in df.columns:
        pet = df["pet"].dropna()
        qa["pet_mean"] = float(pet.mean())
        qa["pet_median"] = float(pet.median())
        qa["pet_p95"] = float(pet.quantile(0.95))
        qa["pet_p99"] = float(pet.quantile(0.99))
        if qa["pet_p99"] < 2.0:
            qa["warnings"].append(
                "PET distribution is highly compressed (p99 < 2.0s); inspect conflict triggering."
            )
        if qa["pet_mean"] < 1.0:
            qa["warnings"].append(
                "Mean PET < 1.0s; verify whether the site is extremely conflict-dense or thresholds are too aggressive."
            )

    if max_frames is not None:
        qa["warnings"].append(f"Run used max_frames={max_frames}; may represent a truncated/debug sample.")

    return qa



def run_detector_benchmark(args: argparse.Namespace, paths: WorkflowPaths, video_path: Path) -> Dict[str, Any]:
    import copy
    import time

    benchmark_results: Dict[str, Any] = {}
    detectors = args.benchmark_detectors_list or ["rtdetr", "sam3"]

    for det in detectors:
        bench_args = copy.deepcopy(args)
        bench_args.detector = det
        bench_args.max_frames = args.benchmark_max_frames
        bench_args.skip_summary = False

        out_csv = paths.output_dir / f"{video_path.stem}_{det}_benchmark_pet.csv"
        t0 = time.perf_counter()

        try:
            run_pet_extraction(bench_args, paths, video_path, out_csv)
            elapsed = time.perf_counter() - t0
            qa = evaluate_pet_quality(out_csv, max_frames=bench_args.max_frames)

            benchmark_results[det] = {
                "status": "ok",
                "elapsed_sec": round(elapsed, 3),
                "pet_csv": str(out_csv),
                "qa": qa,
            }
        except Exception as exc:
            benchmark_results[det] = {
                "status": "failed",
                "error": str(exc),
            }

    return benchmark_results

def run_ground_truth_evaluation(args: argparse.Namespace, paths: WorkflowPaths, pred_csv: Path) -> Dict[str, Any]:
    logger.info("Running ground-truth comparison...")
    out_dir = paths.validation_dir / "ground_truth"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.ground_truth_csv:
        raise ValueError("Ground-truth evaluation requested but --ground-truth-csv not provided.")

    gt_csv = (paths.repo_root / args.ground_truth_csv).resolve()

    try:
        from analysis.validation.ground_truth_metrics import evaluate_pet_events
    except Exception as exc:
        logger.warning("Ground-truth metrics module unavailable.")
        return {"status": "requested_but_module_missing", "error": str(exc)}

    result = evaluate_pet_events(
        pred_csv=str(pred_csv),
        gt_csv=str(gt_csv),
        output_dir=str(out_dir),
        time_tolerance=args.match_time_tolerance,
        frame_tolerance=args.match_frame_tolerance,
        iou_threshold=args.match_iou_threshold,
    )
    logger.info("Ground-truth evaluation outputs: %s", result)
    return result


def run_significance_tests(args: argparse.Namespace, paths: WorkflowPaths, pet_csv: Path) -> Dict[str, Any]:
    logger.info("Running significance tests...")
    out_dir = paths.validation_dir / "significance"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.compare_csvs:
        raise ValueError("Significance testing requested but --compare-csvs not provided.")

    compare_csvs = [(paths.repo_root / p).resolve() for p in args.compare_csvs]

    try:
        from analysis.validation.significance_testing import compare_pet_distributions
    except Exception as exc:
        logger.warning("Significance testing module unavailable.")
        return {"status": "requested_but_module_missing", "error": str(exc)}

    result = compare_pet_distributions(
        reference_csv=str(pet_csv),
        comparison_csvs=[str(p) for p in compare_csvs],
        output_dir=str(out_dir),
        test=args.significance_test,
        n_boot=args.n_boot,
        n_perm=args.n_perm,
        effect_size=args.effect_size,
        alpha=args.alpha,
    )
    logger.info("Significance testing outputs: %s", result)
    return result


def run_diffusion_training(args: argparse.Namespace, paths: WorkflowPaths, pet_csv: Path) -> Dict[str, str]:
    logger.info("Running diffusion training...")
    from traffic_diffusion.train_trajectory_diffusion import main as train_main

    out_dir = (paths.repo_root / args.diffusion_output_dir).resolve() if args.diffusion_output_dir else paths.diffusion_dir
    argv = ["--csv-path", str(pet_csv), "--epochs", str(args.diffusion_epochs), "--output-dir", str(out_dir)]

    if args.diffusion_batch_size is not None:
        argv += ["--batch-size", str(args.diffusion_batch_size)]
    if args.diffusion_lr is not None:
        argv += ["--lr", str(args.diffusion_lr)]
    if args.diffusion_val_split is not None:
        argv += ["--val-split", str(args.diffusion_val_split)]
    if args.diffusion_seed is not None:
        argv += ["--seed", str(args.diffusion_seed)]

    train_main(argv)
    return {"train_output_dir": str(out_dir)}


def run_diffusion_cross_validation(args: argparse.Namespace, paths: WorkflowPaths, pet_csv: Path) -> Dict[str, Any]:
    logger.info("Running diffusion cross-validation...")
    out_dir = paths.validation_dir / "cross_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from analysis.validation.cross_validation import run_diffusion_cv
    except Exception as exc:
        logger.warning("Cross-validation module unavailable.")
        return {"status": "requested_but_module_missing", "error": str(exc)}

    result = run_diffusion_cv(
        csv_path=str(pet_csv),
        output_dir=str(out_dir),
        folds=args.cv_folds,
        repeats=args.cv_repeats,
        epochs=args.diffusion_epochs,
        batch_size=args.diffusion_batch_size,
        lr=args.diffusion_lr,
        seed=args.diffusion_seed,
    )
    logger.info("Cross-validation outputs: %s", result)
    return result


def run_diffusion_eval(args: argparse.Namespace, paths: WorkflowPaths, pet_csv: Path) -> Dict[str, str]:
    logger.info("Running diffusion safety evaluation...")
    from analysis.safety_eval_diffusion import main as eval_main

    out_dir = (paths.repo_root / args.eval_output_dir).resolve() if args.eval_output_dir else paths.output_dir
    argv = ["--csv-path", str(pet_csv), "--output-dir", str(out_dir)]

    if args.eval_checkpoint:
        argv += ["--checkpoint", str((paths.repo_root / args.eval_checkpoint).resolve())]
    if args.eval_num_samples is not None:
        argv += ["--num-samples", str(args.eval_num_samples)]

    eval_main(argv)
    return {
        "safety_events_csv": str(out_dir / "safety_events_diffusion_model.csv"),
        "safety_summary_csv": str(out_dir / "safety_eval_diffusion_summary.csv"),
    }


def save_manifest(paths: WorkflowPaths, args: argparse.Namespace, result: WorkflowResult) -> Path:
    manifest_path = paths.logs_dir / "research_run_manifest.json"
    payload = {
        "version": VERSION,
        "args": vars(args),
        "paths": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(paths).items()},
        "result": asdict(result),
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved manifest -> %s", manifest_path)
    return manifest_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NNDS research workflow orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--video", default=None)
    parser.add_argument("--videos-list", default=None)
    parser.add_argument("--bev-config", default="configs/bev_config.json")
    parser.add_argument("--grid-config", default="configs/GITI_grid_config.json")
    parser.add_argument("--gate-config", default="configs/gate_config.yaml")
    parser.add_argument("--sam3-weights", default="sam3.pt")
    parser.add_argument("--rtdetr-weights", default="rtdetr-l.pt")
    parser.add_argument(
        "--detector",
        choices=["sam3", "rtdetr"],
        default="sam3",
        help="Detection backend: use rtdetr for routine runs; use sam3 for heavy concept-aware segmentation.",
    )
    parser.add_argument("--imgsz", type=int, default=None)

    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--pet-csv", default="outputs/petevents_bev.csv")
    parser.add_argument("--summary-dir", default="analysis_results")
    parser.add_argument("--gate-dir", default="outputs/gates")
    parser.add_argument("--diffusion-dir", default="outputs/diffusion")
    parser.add_argument("--validation-dir", default="outputs/validation")
    parser.add_argument("--logs-dir", default="outputs/logs")

    parser.add_argument("--pet-threshold", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")

    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--run-gate-count", action="store_true")
    parser.add_argument("--train-diffusion", action="store_true")
    parser.add_argument("--eval-diffusion", action="store_true")
    parser.add_argument("--cross-validate-diffusion", action="store_true")
    parser.add_argument("--run-ground-truth-eval", action="store_true")
    parser.add_argument("--run-significance-tests", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--benchmark-detectors", action="store_true")
    parser.add_argument("--benchmark-max-frames", type=int, default=30)
    parser.add_argument("--benchmark-detectors-list", nargs="*", default=["sam3"])

    parser.add_argument("--conflict-col", default="conflict_type")
    parser.add_argument("--baseline-csv", default=None)
    parser.add_argument("--critical", type=float, default=1.0)
    parser.add_argument("--serious", type=float, default=None)
    parser.add_argument("--moderate", type=float, default=3.0)
    parser.add_argument("--summary-format", choices=["json", "csv"], default="json")
    parser.add_argument("--no-risk-buckets", action="store_true")

    parser.add_argument("--ground-truth-csv", default=None)
    parser.add_argument("--match-time-tolerance", type=float, default=0.5)
    parser.add_argument("--match-frame-tolerance", type=int, default=5)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)

    parser.add_argument("--compare-csvs", nargs="*", default=None)
    parser.add_argument("--significance-test", choices=["bootstrap", "permutation"], default="bootstrap")
    parser.add_argument("--effect-size", choices=["cohen_d", "cliffs_delta", "median_shift"], default="cliffs_delta")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-perm", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=0.05)

    parser.add_argument("--gate-detector-module", default=None)
    parser.add_argument("--gate-min-confidence", type=float, default=0.25)
    parser.add_argument("--draw-gate-tracks", action="store_true")
    parser.add_argument("--gate-visual-debug", action="store_true")

    parser.add_argument("--diffusion-epochs", type=int, default=100)
    parser.add_argument("--diffusion-batch-size", type=int, default=None)
    parser.add_argument("--diffusion-lr", type=float, default=None)
    parser.add_argument("--diffusion-output-dir", default=None)
    parser.add_argument("--diffusion-val-split", type=float, default=None)
    parser.add_argument("--diffusion-seed", type=int, default=None)

    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=1)

    parser.add_argument("--eval-checkpoint", default=None)
    parser.add_argument("--eval-num-samples", type=int, default=None)
    parser.add_argument("--eval-output-dir", default=None)

    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args(argv)


def run_single_video(args: argparse.Namespace, paths: WorkflowPaths, video_path: Path) -> Dict[str, Any]:
    result = WorkflowResult(
        warnings=[],
        notes=[],
        qa={},
        validation_outputs={},
    )
    thresholds = resolve_thresholds(args)

    stem = video_path.stem
    pet_csv = paths.pet_csv if paths.video == video_path else paths.output_dir / f"{stem}_petevents_bev.csv"
    summary_dir = paths.summary_dir if paths.video == video_path else paths.summary_dir / stem
    summary_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_extraction:
        run_pet_extraction(args, paths, video_path, pet_csv)
        result.extraction_ran = True
    else:
        result.notes.append("Extraction skipped; existing PET CSV reused.")

    result.pet_csv = str(pet_csv)

    if not args.skip_summary:
        result.summary_exports = run_pet_summary(args, pet_csv, summary_dir, thresholds)
        result.summary_ran = True
    else:
        result.notes.append("PET summary skipped.")

    qa = evaluate_pet_quality(pet_csv, max_frames=args.max_frames)
    result.qa = qa
    result.warnings.extend(qa.get("warnings", []))

    if args.run_gate_count:
        result.notes.append("Gate counting hook retained; add project-specific implementation if needed.")

    if args.train_diffusion:
        result.diffusion_outputs = result.diffusion_outputs or {}
        result.diffusion_outputs.update(run_diffusion_training(args, paths, pet_csv))
        result.diffusion_train_ran = True

    if args.cross_validate_diffusion:
        result.validation_outputs["cross_validation"] = run_diffusion_cross_validation(args, paths, pet_csv)
        result.cross_validation_ran = True

    if args.eval_diffusion:
        result.diffusion_outputs = result.diffusion_outputs or {}
        result.diffusion_outputs.update(run_diffusion_eval(args, paths, pet_csv))
        result.diffusion_eval_ran = True

    if args.run_ground_truth_eval:
        result.validation_outputs["ground_truth"] = run_ground_truth_evaluation(args, paths, pet_csv)
        result.ground_truth_eval_ran = True

    if args.run_significance_tests:
        result.validation_outputs["significance"] = run_significance_tests(args, paths, pet_csv)
        result.significance_tests_ran = True

    return asdict(result)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    setup_logging(args.verbose)
    paths = build_paths(args)
    ensure_dirs(paths)
    validate_required_inputs(paths, args.detector, args.skip_extraction)

    try:
        if args.videos_list:
            videos = [((paths.repo_root / v).resolve()) for v in load_video_list(paths.videos_list)]
        else:
            videos = [paths.video]

        all_results: Dict[str, Any] = {"version": VERSION, "videos": {}}

        for video_path in videos:
            if not video_path.exists():
                raise FileNotFoundError(f"Video not found: {video_path}")

            video_result = run_single_video(args, paths, video_path)

            if args.benchmark_detectors:
                video_result["detector_benchmark"] = run_detector_benchmark(args, paths, video_path)

            all_results["videos"][str(video_path)] = video_result

        manifest_result = WorkflowResult(
            extraction_ran=any(v["extraction_ran"] for v in all_results["videos"].values()),
            summary_ran=any(v["summary_ran"] for v in all_results["videos"].values()),
            gate_count_ran=any(v["gate_count_ran"] for v in all_results["videos"].values()),
            diffusion_train_ran=any(v["diffusion_train_ran"] for v in all_results["videos"].values()),
            diffusion_eval_ran=any(v["diffusion_eval_ran"] for v in all_results["videos"].values()),
            cross_validation_ran=any(v["cross_validation_ran"] for v in all_results["videos"].values()),
            ground_truth_eval_ran=any(v["ground_truth_eval_ran"] for v in all_results["videos"].values()),
            significance_tests_ran=any(v["significance_tests_ran"] for v in all_results["videos"].values()),
            notes=["Per-video outputs are stored under the `videos` key."],
            warnings=[],
            qa={"thresholds": resolve_thresholds(args)},
        )

        manifest_path = save_manifest(paths, args, manifest_result)
        all_results["manifest"] = str(manifest_path)

        sys.stdout.flush()
        print(json.dumps(all_results, indent=2))
        sys.stdout.flush()

    except Exception as exc:
        logger.error("Research workflow failed: %s", exc)
        logger.debug(traceback.format_exc())
        sys.stderr.write(json.dumps({"error": str(exc)}, indent=2) + "\n")
        sys.stderr.flush()
        raise


if __name__ == "__main__":
    main()
