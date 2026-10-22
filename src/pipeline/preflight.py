"""Preflight validation for the traffic analyzer pipeline.

Run from CLI:
    python -m src.pipeline.preflight --video data/sample_data/x.mp4 \
        --uvh-model data/models/uvh26.pt

Or import:
    from src.pipeline.preflight import run_checks
    report = run_checks(video=..., uvh_model=...)
    if not report.ok:
        for p in report.problems:
            print(p)
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

MIN_VIDEO_BYTES = 100_000
MIN_WEIGHT_BYTES = 1_000_000


@dataclass
class PreflightReport:
    checks: list = field(default_factory=list)
    problems: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def add(self, name: str, ok: bool, detail: str, fix: str = "") -> None:
        self.checks.append({"check": name, "ok": ok, "detail": detail, "fix": fix})
        if not ok:
            self.problems.append(f"{name}: {detail}" + (f"  fix: {fix}" if fix else ""))

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "checks": self.checks,
            "problems": self.problems,
        }


def _check_file(r: PreflightReport, path: str, name: str, min_bytes: int, fix: str) -> None:
    p = Path(path)
    if not p.exists():
        r.add(name, False, f"missing: {path}", fix)
        return
    size = p.stat().st_size
    if size < min_bytes:
        r.add(
            name,
            False,
            f"{path} is {size} bytes (< {min_bytes}); likely an LFS pointer or corrupt file",
            "run `git lfs pull` to fetch real data",
        )
        return
    r.add(name, True, f"{path} ({size / 1e6:.1f} MB)")


def _parse_config_text(text: str, suffix: str) -> None:
    """Parse config text as JSON or YAML depending on file extension."""
    suffix = suffix.lower()
    if suffix in (".yaml", ".yml"):
        import yaml as _yaml

        _yaml.safe_load(text)
        return
    if suffix == ".json":
        json.loads(text)
        return
    # Unknown extension - try JSON then YAML.
    try:
        json.loads(text)
    except json.JSONDecodeError:
        import yaml as _yaml

        _yaml.safe_load(text)


def _check_config(r: PreflightReport, path: str | None, name: str) -> None:
    if path is None:
        r.add(name, True, "not provided (optional)")
        return
    p = Path(path)
    if not p.exists():
        r.add(name, False, f"missing: {path}", "check the config path or omit the flag")
        return
    try:
        _parse_config_text(p.read_text(), p.suffix)
        r.add(name, True, f"{path} parsed")
    except Exception as e:
        r.add(
            name,
            False,
            f"{path} not valid JSON/YAML: {e}",
            "restore from git: git checkout <sha> -- <path>",
        )


def _check_device(r: PreflightReport, requested: str) -> None:
    try:
        import torch
    except ImportError:
        r.add("torch available", False, "torch not installed", "pip install -r requirements.txt")
        return
    r.add("torch available", True, f"torch {torch.__version__}")
    cuda = torch.cuda.is_available()
    try:
        mps = torch.backends.mps.is_available()
    except (AttributeError, ImportError):
        mps = False

    if requested == "cuda" and not cuda:
        r.add(
            "device cuda",
            False,
            "cuda requested but not available",
            "use --device auto or --device cpu",
        )
    elif requested == "mps" and not mps:
        # Pre-existing bug: only "cuda" was validated; every other device
        # name (including mps on Linux/Windows) passed silently.
        r.add(
            "device mps",
            False,
            "mps requested but not available (requires macOS + Apple Silicon)",
            "use --device auto or --device cpu",
        )
    elif requested == "auto":
        r.add("device auto", True, "cuda" if cuda else ("mps" if mps else "cpu"))
    elif requested == "cpu":
        r.add("device cpu", True, "ok")
    else:
        r.add(f"device {requested}", True, "ok")


def _check_ffmpeg(r: PreflightReport) -> None:
    """Report ffmpeg availability as informational.

    The fused detector (and the yolo-cpu and sam3 backends) all read video
    through OpenCV, not ffmpeg. A missing ffmpeg does not block any of them,
    so this check is a warning rather than a failure. Previously a missing
    ffmpeg turned an otherwise-healthy report into `ok=False`, so
    `--self-check` exited 1 on machines that can run the pipeline fine.
    """
    path = shutil.which("ffmpeg")
    if path is None:
        r.add(
            "ffmpeg on PATH",
            True,
            "not found (optional - OpenCV handles video reads)",
            "install only if using a backend that shells out to ffmpeg",
        )
    else:
        r.add("ffmpeg on PATH", True, path)


def _check_outdir(r: PreflightReport, path: str | None) -> None:
    """Verify the output path parent directory is writable.

    Two pre-existing bugs fixed here:

    1. ``p.parent.touch(exist_ok=True)`` called touch() on a directory, which
       is not a write test. The new implementation writes a temp file inside
       the directory, then deletes it.
    2. ``mkdir()`` was outside the ``try`` block, so permission errors there
       propagated as exceptions instead of reported failed checks.

    Paths containing ``..`` are rejected (SonarCloud path-traversal finding).
    A legitimate ``--out-csv`` is either a relative path within the working
    tree or an absolute path the caller fully controls.
    """
    if not path:
        r.add("output dir writable", True, "no output path provided; skipped")
        return
    p = Path(path)
    if ".." in p.parts:
        r.add(
            "output dir writable",
            False,
            f"refusing path with '..': {path}",
            "use a relative path without '..' (e.g. outputs/pet.csv) "
            "or an absolute path to a writable directory",
        )
        return
    parent = p.parent if p.parent != Path("") else Path(".")
    try:
        parent.mkdir(parents=True, exist_ok=True)
        fd, probe = tempfile.mkstemp(prefix=".preflight-probe-", dir=str(parent))
        os.close(fd)
        os.unlink(probe)
        r.add("output dir writable", True, str(parent))
    except Exception as e:
        r.add("output dir writable", False, str(e), "check directory permissions")


def _check_ultralytics(r: PreflightReport) -> None:
    try:
        import ultralytics

        r.add("ultralytics available", True, f"ultralytics {ultralytics.__version__}")
    except ImportError:
        r.add("ultralytics available", False, "not installed", "pip install ultralytics")


def run_checks(
    video: str | None = None,
    uvh_model: str | None = None,
    yolo_weights: str | None = None,
    coco_person_model: str | None = None,
    rtdetr_weights: str | None = None,
    sam3_weights: str | None = None,
    bev_config: str | None = None,
    grid_config: str | None = None,
    gate_config: str | None = None,
    out_csv: str = "outputs/petevents_bev.csv",
    device: str = "auto",
    detector: str = "uvh-coco-fused",
) -> PreflightReport:
    r = PreflightReport()

    if video:
        _check_file(
            r,
            video,
            "video file",
            MIN_VIDEO_BYTES,
            "ensure LFS is pulled (git lfs pull) or place the file at this path",
        )
    else:
        r.add("video file", True, "not provided (demo mode?)")

    if detector in ("uvh-coco-fused", "uvh-coco"):
        if uvh_model:
            _check_file(
                r, uvh_model, "uvh model", MIN_WEIGHT_BYTES, "download from HF: iisc-aim/UVH-26"
            )
        # `uvh-coco-fused` uses --coco-person-model, not --yolo-weights.
        # They default to the same file, but a user who overrides only one
        # was getting a check against the wrong path. Prefer
        # coco_person_model when set; keep yolo_weights as a fallback for
        # callers using the older argument.
        if coco_person_model:
            _check_file(
                r,
                coco_person_model,
                "coco person model",
                MIN_WEIGHT_BYTES,
                "ultralytics will auto-download on first use",
            )
        elif yolo_weights:
            _check_file(
                r,
                yolo_weights,
                "yolo weights",
                MIN_WEIGHT_BYTES,
                "ultralytics will auto-download on first use",
            )
    if detector == "yolo-cpu":
        # yolo-cpu requires weights (run_video_to_pet raises FileNotFoundError
        # if the file is missing). Use the same default that run_pipeline
        # supplies so a `--self-check` with no explicit --yolo-weights still
        # verifies the file the run will actually use.
        weights = yolo_weights or "data/models/yolo11n.pt"
        _check_file(
            r,
            weights,
            "yolo weights",
            MIN_WEIGHT_BYTES,
            "download yolo11n.pt from ultralytics assets",
        )
    if detector == "rtdetr" and rtdetr_weights:
        _check_file(
            r,
            rtdetr_weights,
            "rtdetr weights",
            MIN_WEIGHT_BYTES,
            "download rtdetr-l.pt from ultralytics assets",
        )
    if detector == "sam3" and sam3_weights:
        _check_file(
            r,
            sam3_weights,
            "sam3 weights",
            MIN_WEIGHT_BYTES,
            "request access at huggingface.co/facebookresearch/sam3",
        )

    for name, path in (
        ("bev config", bev_config),
        ("grid config", grid_config),
        ("gate config", gate_config),
    ):
        _check_config(r, path, name)

    _check_device(r, device)
    _check_ffmpeg(r)
    _check_ultralytics(r)
    _check_outdir(r, out_csv)
    return r


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Preflight checks for traffic analyzer")
    ap.add_argument("--video")
    ap.add_argument("--uvh-model")
    ap.add_argument("--yolo-weights")
    ap.add_argument("--coco-person-model")
    ap.add_argument("--rtdetr-weights")
    ap.add_argument("--sam3-weights")
    ap.add_argument("--bev-config")
    ap.add_argument("--grid-config")
    ap.add_argument("--gate-config")
    ap.add_argument("--out-csv", default="outputs/petevents_bev.csv")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--detector", default="uvh-coco-fused")
    args = ap.parse_args()

    report = run_checks(
        video=args.video,
        uvh_model=args.uvh_model,
        yolo_weights=args.yolo_weights,
        coco_person_model=args.coco_person_model,
        rtdetr_weights=args.rtdetr_weights,
        sam3_weights=args.sam3_weights,
        bev_config=args.bev_config,
        grid_config=args.grid_config,
        gate_config=args.gate_config,
        out_csv=args.out_csv,
        device=args.device,
        detector=args.detector,
    )
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
