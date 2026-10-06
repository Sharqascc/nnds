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
import shutil
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


def _check_config(r: PreflightReport, path: str | None, name: str) -> None:
    if path is None:
        r.add(name, True, "not provided (optional)")
        return
    p = Path(path)
    if not p.exists():
        r.add(name, False, f"missing: {path}", "check the config path or omit the flag")
        return
    try:
        json.loads(p.read_text())
        r.add(name, True, f"{path} parsed")
    except Exception as e:
        r.add(
            name,
            False,
            f"{path} not valid JSON: {e}",
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
    if requested == "cuda" and not cuda:
        r.add(
            "device cuda",
            False,
            "cuda requested but not available",
            "use --device auto or --device cpu",
        )
    elif requested == "auto":
        r.add("device auto", True, "cuda" if cuda else "cpu")
    else:
        r.add(f"device {requested}", True, "ok")


def _check_ffmpeg(r: PreflightReport) -> None:
    if shutil.which("ffmpeg") is None:
        r.add("ffmpeg on PATH", False, "ffmpeg not found", "apt-get install -y ffmpeg")
    else:
        r.add("ffmpeg on PATH", True, shutil.which("ffmpeg"))


def _check_outdir(r: PreflightReport, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        p.parent.touch(exist_ok=True)
        r.add("output dir writable", True, str(p.parent))
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
        if yolo_weights:
            _check_file(
                r,
                yolo_weights,
                "yolo weights",
                MIN_WEIGHT_BYTES,
                "ultralytics will auto-download on first use",
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
