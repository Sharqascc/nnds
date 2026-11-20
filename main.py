#!/usr/bin/env python3
"""Reproducible main entry point for NNDS in Colab."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parent

PROJECT_NAME = "nnds"
PROJECT_MODULE = "pipeline"
CLI_MODULE = "pipeline.traffic_analyzer"
CLI_COMMAND = "nnds"

RUNTIME_REQUIREMENTS = (
    "torch>=2.14.1,<2.15",
    "torchvision>=0.29.1,<0.30",
    "ultralytics>=8.4.178,<8.5",
    "opencv-python>=5.0.0,<5.1",
    "numpy>=2.5.4,<2.6",
    "pandas>=3.0.6,<3.1",
    "scipy>=1.18.1,<1.19",
    "matplotlib>=3.11.2,<3.12",
    "tqdm>=4.70.1,<4.71",
    "PyYAML>=6.0.3,<6.1",
    "openvino>=2026.4.1,<2026.5",
    "lap>=0.5.13,<0.6",
    "scikit-learn>=1.9.1,<1.10",
)


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _importable(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def install_requirements(*, install_dev: bool = True) -> None:
    extras = ".[dev]"

    print(f"[1/5] Installing requirements: pip install -e {extras}")

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
        cwd=REPO_ROOT,
        check=True,
    )

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", extras],
        cwd=REPO_ROOT,
        check=True,
    )


def verify_installation() -> None:
    print("[2/5] Verifying installation")

    missing = []

    for requirement in RUNTIME_REQUIREMENTS:
        package = requirement.split(">=")[0].strip()

        if _version(package) is None:
            missing.append(requirement)

    if missing:
        raise RuntimeError("Missing runtime requirements: " + ", ".join(missing))

    if _version(PROJECT_NAME) is None:
        raise RuntimeError(f"Project not installed: {PROJECT_NAME}")

    if not _importable(PROJECT_MODULE):
        raise RuntimeError(f"Module not importable: {PROJECT_MODULE}")

    if not _importable(CLI_MODULE):
        raise RuntimeError(f"CLI module not importable: {CLI_MODULE}")

    test_requirements = (
        "pytest",
        "pytest-cov",
        "pytest-xdist",
        "pytest-timeout",
        "pytest-mock",
        "hypothesis",
    )

    missing_test_requirements = [
        requirement for requirement in test_requirements if _version(requirement) is None
    ]

    if missing_test_requirements:
        raise RuntimeError("Missing test requirements: " + ", ".join(missing_test_requirements))

    print(
        f"      Verified {PROJECT_NAME} {_version(PROJECT_NAME)}; "
        "all runtime and test requirements installed."
    )


def run_main_task(forwarded_args: Sequence[str]) -> None:
    print("[3/5] Running nnds")

    if forwarded_args and forwarded_args[0] == "--":
        forwarded_args = forwarded_args[1:]

    if not forwarded_args:
        print(
            "      No task arguments supplied; the canonical GITI workflow "
            "will run in the remaining-workflow stage."
        )
        return

    command = [
        sys.executable,
        "-m",
        CLI_MODULE,
        *forwarded_args,
    ]

    print("      " + " ".join(command))

    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)

    if completed.returncode != 0:
        raise RuntimeError(f"nnds failed with exit code {completed.returncode}.")


def run_stage(name: str, command: list[str], optional_inputs: Sequence[Path] = ()) -> dict:
    """Run one workflow stage with live output and elapsed-time updates."""

    missing = [path for path in optional_inputs if not path.exists()]
    if missing:
        print(
            f"      Skipping {name}: missing input(s): " + ", ".join(str(path) for path in missing)
        )
        return {
            "name": name,
            "status": "skipped",
            "missing_inputs": [str(path) for path in missing],
        }

    printable_command = " ".join(command)
    print(f"      Running {name}: {printable_command}")
    print(f"      Started: {time.strftime('%H:%M:%S')}")

    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    last_heartbeat = time.monotonic()

    assert process.stdout is not None
    for line in process.stdout:
        sys.stdout.write(f"      | {line}")
        sys.stdout.flush()

        now = time.monotonic()
        if now - last_heartbeat >= 30:
            elapsed = int(now - started)
            print(f"      [{elapsed // 60:02d}:{elapsed % 60:02d}] Still running: {name}")
            sys.stdout.flush()
            last_heartbeat = now

    return_code = process.wait()
    duration = time.monotonic() - started

    print(f"      Finished {name}: exit code {return_code}, duration {duration:.1f} seconds")

    if return_code != 0:
        raise RuntimeError(f"{name} failed with exit code {return_code}.")

    return {
        "name": name,
        "status": "completed",
        "command": command,
        "returncode": return_code,
        "duration_seconds": round(duration, 3),
    }


def ensure_lfs_video(video_path: Path) -> None:
    """Ensure a Git LFS video is downloaded and is not just a pointer file."""

    if not video_path.exists():
        raise FileNotFoundError(f"Configured NNDS video does not exist: {video_path}")

    # Git LFS pointer files are tiny text files beginning with this marker.
    try:
        head = video_path.read_bytes()[:200]
    except OSError as exc:
        raise RuntimeError(f"Could not read video file {video_path}: {exc}")

    is_pointer = head.startswith(b"version https://git-lfs.github.com/spec/v1")

    if video_path.stat().st_size < 1024 or is_pointer:
        print(f"      Video appears to be a Git LFS pointer; pulling: {video_path}")
        commands = [
            ["apt-get", "update", "-qq"],
            ["apt-get", "install", "-y", "-qq", "git-lfs"],
            ["git", "lfs", "install"],
            ["git", "lfs", "pull", "--include", str(video_path.relative_to(REPO_ROOT))],
        ]

        for command in commands:
            completed = subprocess.run(
                command,
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
            )
            if completed.stdout:
                print(completed.stdout.rstrip())
            if completed.stderr:
                print(completed.stderr.rstrip(), file=sys.stderr)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Git LFS command failed with exit code "
                    f"{completed.returncode}: {' '.join(command)}"
                )

    # Recheck after pulling.
    try:
        head = video_path.read_bytes()[:200]
    except OSError as exc:
        raise RuntimeError(f"Could not read video file {video_path}: {exc}")

    if video_path.stat().st_size < 1024 or head.startswith(
        b"version https://git-lfs.github.com/spec/v1"
    ):
        raise RuntimeError(
            f"Git LFS video was not downloaded correctly: {video_path}. "
            "Ensure the file is stored with Git LFS and that you have access "
            "to its LFS storage."
        )

    print(f"      Video ready: {video_path} ({video_path.stat().st_size / 1024 / 1024:.1f} MB)")


def run_remaining_tasks(forwarded_args: Sequence[str] | None = None) -> None:
    """Run the complete NNDS system and paper workflow."""

    print("[4/6] Running remaining project workflow stages")

    if forwarded_args is None:
        forwarded_args = []

    if forwarded_args and forwarded_args[0] == "--":
        forwarded_args = forwarded_args[1:]

    pipeline_args = list(forwarded_args)

    default_video = REPO_ROOT / "data" / "sample_data" / "GITI_traffic_video.mp4"

    if not pipeline_args:
        ensure_lfs_video(default_video)
        pipeline_args = [
            "--video",
            str(default_video),
            "--detector",
            "uvh-coco-fused",
            "--video-source",
            "GITI",
        ]

    pipeline_command = [
        sys.executable,
        "scripts/run_pipeline.py",
        *pipeline_args,
    ]

    pet_csv = REPO_ROOT / "outputs" / "pet_events" / "bev.csv"
    sample_video = default_video
    annotator_a = REPO_ROOT / "paper" / "annotations" / "workbook_annotator_A.csv"
    annotator_b = REPO_ROOT / "paper" / "annotations" / "workbook_annotator_B.csv"

    results: list[dict] = []

    results.append(
        run_stage(
            "Detection, tracking, BEV, gate counting, and PET extraction",
            pipeline_command,
        )
    )

    if pet_csv.exists():
        results.append(
            run_stage(
                "Validate NNDS pipeline outputs",
                [
                    sys.executable,
                    "scripts/validate_outputs.py",
                    "--pet",
                    str(pet_csv),
                    "--detections",
                    str(pet_csv.with_name(pet_csv.stem + "_detections.csv")),
                    "--video-frames",
                    "1838",
                ],
            )
        )
    else:
        print("      Skipping output validation: PET CSV was not produced.")
        results.append(
            {
                "name": "Validate NNDS pipeline outputs",
                "status": "skipped",
                "missing_inputs": [str(pet_csv)],
            }
        )

    paper_stages = [
        (
            "Paper pair analysis",
            [sys.executable, "paper/analysis/pair_analysis.py"],
            [],
        ),
        (
            "Paper statistical tests",
            [sys.executable, "paper/analysis/statistical_tests.py"],
            [],
        ),
        (
            "Generate false-positive taxonomy",
            [sys.executable, "paper/analysis/generate_fp_taxonomy.py"],
            [],
        ),
        (
            "Validate paper ground truth",
            [sys.executable, "paper/analysis/gt_validation.py"],
            [annotator_a, annotator_b],
        ),
        (
            "Build false-positive clips",
            [sys.executable, "paper/analysis/build_fp_clips.py"],
            [sample_video],
        ),
    ]

    for name, command, optional_inputs in paper_stages:
        results.append(
            run_stage(
                name,
                command,
                optional_inputs,
            )
        )

    output_path = REPO_ROOT / "outputs" / "workflow_stage_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))

    print(f"      Wrote {output_path}")


def write_run_manifest() -> None:
    print("[5/5] Writing run manifest")

    manifest_path = REPO_ROOT / "outputs" / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "project": PROJECT_NAME,
        "project_version": _version(PROJECT_NAME),
        "python": sys.version,
        "cli_command": CLI_COMMAND,
        "cli_module": CLI_MODULE,
    }

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"      Wrote {manifest_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_colab.py",
        description=(
            "Reproducible Colab entry point for NNDS. Installs requirements, "
            "verifies the environment, runs the nnds CLI, runs its built-in "
            "demo, and writes a run manifest."
        ),
    )

    parser.add_argument(
        "--install-dev",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Install development dependencies (default: enabled).",
    )
    parser.add_argument(
        "--scientific-tests",
        action="store_true",
        help="Run reproducible PET scientific-validation tests and exit.",
    )
    parser.add_argument(
        "--input",
        default="outputs/combined_screened_simplified.csv",
        help="PET CSV used by --scientific-tests.",
    )
    parser.add_argument(
        "--output-dir",
        default="paper/results",
        help="Output directory used by --scientific-tests.",
    )
    parser.add_argument(
        "cli_args",
        nargs="*",
        help="Arguments forwarded directly to the nnds CLI.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    src_root = REPO_ROOT / "src"

    for path in (str(REPO_ROOT), str(src_root)):
        if path not in sys.path:
            sys.path.insert(0, path)

    os.environ["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(src_root), os.environ.get("PYTHONPATH", "")) if path
    )

    os.chdir(REPO_ROOT)

    install_requirements(install_dev=args.install_dev)
    verify_installation()

    if args.scientific_tests:
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "run_scientific_tests.py"),
                "--input",
                args.input,
                "--output-dir",
                args.output_dir,
            ],
            check=False,
        )
        return result.returncode

    forwarded_args = args.cli_args

    # A frame-limited invocation is a smoke test: run only the analyzer.
    if any(arg == "--max-frames" or arg.startswith("--max-frames=") for arg in forwarded_args):
        run_main_task(forwarded_args)
        return 0

    run_main_task(forwarded_args)
    run_remaining_tasks()
    write_run_manifest()

    print("[done] Main entry-point workflow completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
