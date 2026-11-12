"""NNDS main entry point (MEP).

Console command:
    nnds

Implementation:
    src/pipeline/traffic_analyzer.py:main

This module is the primary user-facing CLI for NNDS. It detects vehicles and
pedestrians, tracks them, projects detections into bird's-eye view, and
computes PET safety events.

`nnds-run` (`scripts/run_pipeline.py`) is a separate preflight and dispatch
wrapper around this MEP; it is not the MEP itself.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class WorldPoint:
    t: float
    x: float
    y: float


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class CompleteTrafficAnalyzer:
    """BEV calibration + speed estimation helper."""

    def __init__(self, bev_width: int = 800, bev_height: int = 800):
        self.bev_width = bev_width
        self.bev_height = bev_height
        self.pixel_points: np.ndarray | None = None
        self.world_points_approx: np.ndarray | None = None
        self.homography: np.ndarray | None = None
        self.inv_homography: np.ndarray | None = None
        self.inlier_mask: np.ndarray | None = None
        self.calibration_metrics: dict = {}
        self.bev_x_min: float | None = None
        self.bev_x_max: float | None = None
        self.bev_y_min: float | None = None
        self.bev_y_max: float | None = None
        self.meters_per_pixel_x: float | None = None
        self.meters_per_pixel_y: float | None = None

    def calibrate(
        self,
        pixel_points,
        world_points_approx,
        ransac_threshold: float = 5.0,
        ransac_confidence: float = 0.99,
        ransac_max_iters: int = 5000,
    ):
        pixel_pts = np.array(pixel_points, dtype=np.float32)
        self.pixel_points = pixel_pts
        wpts = np.array(world_points_approx, dtype=np.float32)
        self.world_points_approx = wpts
        H, mask = cv2.findHomography(
            pixel_pts,
            wpts[:, :2],
            cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold,
            confidence=ransac_confidence,
            maxIters=ransac_max_iters,
        )
        if H is None:
            raise RuntimeError("Homography estimation failed")
        self.homography = H
        self.inv_homography = np.linalg.inv(H)
        if mask is not None:
            self.inlier_mask = mask.ravel().astype(bool)
            projected = cv2.perspectiveTransform(
                pixel_pts.reshape(-1, 1, 2), self.homography
            ).reshape(-1, 2)
            errors = np.linalg.norm(projected - wpts[:, :2], axis=1)
            self.calibration_metrics["final_mae"] = float(np.mean(errors))
        else:
            self.inlier_mask = None
            self.calibration_metrics["final_mae"] = 0.0
        return H, mask

    def _calculate_bev_scale(self, safety_margin: float = 0.1):
        if self.world_points_approx is None or self.inlier_mask is None:
            return
        pts = self.world_points_approx[self.inlier_mask][:, :2]
        if len(pts) == 0:
            return
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        x_span = float(x_max - x_min)
        y_span = float(y_max - y_min)
        mx = x_span * safety_margin if x_span > 0 else 1.0
        my = y_span * safety_margin if y_span > 0 else 1.0
        self.bev_x_min = float(x_min - mx)
        self.bev_x_max = float(x_max + mx)
        self.bev_y_min = float(y_min - my)
        self.bev_y_max = float(y_max + my)
        self.meters_per_pixel_x = (self.bev_x_max - self.bev_x_min) / self.bev_width
        self.meters_per_pixel_y = (self.bev_y_max - self.bev_y_min) / self.bev_height

    def pixel_to_world(self, pixel_point):
        if self.homography is None:
            raise RuntimeError("Homography not initialized")
        pixel_h: np.ndarray = np.append(np.array(pixel_point, dtype=np.float32), 1.0).reshape(3, 1)
        world_h = self.homography @ pixel_h
        if abs(world_h[2]) < 1e-9:
            raise ValueError(f"pixel {pixel_point} maps to infinity (world_h[2]={world_h[2]})")
        return (world_h[:2] / world_h[2]).ravel()

    def validate_bev(self):
        if self.homography is None or self.pixel_points is None or self.world_points_approx is None:
            raise RuntimeError("Calibration must be run before BEV validation")
        projected = cv2.perspectiveTransform(
            self.pixel_points.reshape(-1, 1, 2), self.homography
        ).reshape(-1, 2)
        errors = np.linalg.norm(projected - self.world_points_approx[:, :2], axis=1)
        mean_all = float(np.mean(errors))
        if self.inlier_mask is not None and self.inlier_mask.any():
            mean_inliers = float(np.mean(errors[self.inlier_mask]))
        else:
            mean_inliers = mean_all
        rmse = float(np.sqrt(np.mean(errors**2)))
        return {"mean_error_all": mean_all, "mean_error": mean_inliers, "rmse": rmse}

    def estimate_speed(self, pixel_positions, frame_times, fps: float = 30.0):
        # fps is accepted for API compatibility; frame_times is expected
        # to already be in seconds, so fps is not used in the calculation.
        if self.homography is None:
            raise RuntimeError("Homography not initialized")
        pixel_positions = np.asarray(pixel_positions, dtype=np.float32)
        frame_times = np.asarray(frame_times, dtype=np.float32)
        if len(pixel_positions) != len(frame_times):
            raise ValueError("pixel_positions and frame_times must have the same length")
        valid = []
        for p, t in zip(pixel_positions, frame_times, strict=False):
            if np.all(np.isfinite(p)) and np.isfinite(t):
                valid.append((p, float(t)))
        if len(valid) < 5:
            return {"final_speed": 15.0, "speed_std": 2.0, "fallback": True}
        world_pts = [self.pixel_to_world(p) for p, _ in valid]
        times = [t for _, t in valid]
        speeds = []
        for i in range(1, len(world_pts)):
            dist = float(np.linalg.norm(world_pts[i] - world_pts[i - 1]))
            dt = times[i] - times[i - 1]
            if dt <= 0:
                continue
            spd_kmh = (dist / dt) * 3.6
            if 0.5 < spd_kmh < 150.0:
                speeds.append(spd_kmh)
        if len(speeds) < 3:
            return {"final_speed": 15.0, "speed_std": 2.0, "fallback": True}
        return {
            "final_speed": float(np.median(speeds)),
            "speed_std": float(np.std(speeds)),
        }

    def save_calibration(self, path):
        data = {
            "homography": self.homography.tolist() if self.homography is not None else None,
            "bev_bounds": {
                "x_min": self.bev_x_min,
                "x_max": self.bev_x_max,
                "y_min": self.bev_y_min,
                "y_max": self.bev_y_max,
            },
            "calibration_metrics": self.calibration_metrics,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, indent=2))

    def generate_report(self, speed_results):
        return {
            "speed": speed_results,
            "bev_bounds": {
                "x_min": self.bev_x_min,
                "x_max": self.bev_x_max,
                "y_min": self.bev_y_min,
                "y_max": self.bev_y_max,
            },
            "calibration_metrics": self.calibration_metrics,
        }


# ---------------------------------------------------------------------------
# Event helpers
# ---------------------------------------------------------------------------


def _get(event, key, default=None):
    if isinstance(event, dict):
        return event.get(key, default)
    return getattr(event, key, default)


def _get_any(event, *keys, default=None):
    for k in keys:
        v = _get(event, k, None)
        if v is not None:
            return v
    return default


def _parse_track_id(value):
    if value is None:
        return -1
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        if not math.isfinite(f):
            return -1
        return int(f)
    if isinstance(value, str):
        m = re.search(r"(-?\d+)\s*$", value.strip())
        if m:
            return int(m.group(1))
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return -1
    return -1


def _extract_events(result):
    if result is None:
        return []
    if isinstance(result, dict):
        return list(result.get("pet_events") or [])
    events = getattr(result, "pet_events", None)
    return list(events) if events else []


_EVENT_COLUMNS = [
    "pet",
    "frame",
    "track_a",
    "track_b",
    "conflict_type",
    "grid_cell",
    "track_a_entry_frame",
    "track_a_exit_frame",
    "track_b_entry_frame",
    "track_b_exit_frame",
    "track_a_exit_time_sec",
    "track_b_entry_time_sec",
    "track_b_exit_time_sec",
    "world_traj_i",
    "world_traj_j",
    "traj_a_json",
    "traj_b_json",
    "video_source",
    "time_of_day_label",
    "gate_a_entry",
    "gate_b_entry",
]


def _events_to_dataframe(events):
    rows = []
    for event in events:
        row = {
            "pet": _get_any(event, "pet", "PET"),
            "frame": _get_any(event, "frame", "frame_idx"),
            "track_a": _parse_track_id(_get(event, "track_a")),
            "track_b": _parse_track_id(_get(event, "track_b")),
            "conflict_type": _get(event, "conflict_type"),
            "grid_cell": _get(event, "grid_cell"),
            "track_a_entry_frame": _get(event, "track_a_entry_frame"),
            "track_a_exit_frame": _get(event, "track_a_exit_frame"),
            "track_b_entry_frame": _get(event, "track_b_entry_frame"),
            "track_b_exit_frame": _get(event, "track_b_exit_frame"),
            "track_a_exit_time_sec": _get(event, "track_a_exit_time_sec"),
            "track_b_entry_time_sec": _get(event, "track_b_entry_time_sec"),
            "track_b_exit_time_sec": _get(event, "track_b_exit_time_sec"),
            "world_traj_i": _get(event, "world_traj_i"),
            "world_traj_j": _get(event, "world_traj_j"),
            "traj_a_json": _get(event, "traj_a_json"),
            "traj_b_json": _get(event, "traj_b_json"),
            "video_source": _get(event, "video_source"),
            "time_of_day_label": _get(event, "time_of_day_label"),
            "gate_a_entry": _get(event, "gate_a_entry"),
            "gate_b_entry": _get(event, "gate_b_entry"),
        }
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=_EVENT_COLUMNS)
    return pd.DataFrame(rows)


def _write_events_to_csv(events, out_csv_path):
    df = _events_to_dataframe(events)
    if df.empty:
        warnings.warn("No PET events detected", RuntimeWarning, stacklevel=2)
    p = Path(out_csv_path)
    if str(p.parent) not in ("", "."):
        p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return df


# ---------------------------------------------------------------------------
# Detector dispatcher
# ---------------------------------------------------------------------------


def _ensure_src_namespace() -> None:
    """Make `src.<pkg>` importable even though the package is src-layout.

    `pip install -e .` puts `pipeline`, `analysis`, `bev`, ... at the top of
    sys.path (via the .pth file); the repository's internal modules still do
    `from src.pipeline.custom_tracker import ...`. This registers a synthetic
    `src` package whose `__path__` points at wherever those top-level
    packages live, so `import src.pipeline.custom_tracker` resolves cleanly
    after an editable or wheel install.
    """
    import importlib.util
    import sys
    import types
    from pathlib import Path

    existing = sys.modules.get("src")
    if existing is not None and getattr(existing, "_nnds_alias", False):
        return

    parent: Path | None = None
    for candidate in ("pipeline", "analysis", "bev", "core", "utils"):
        spec = importlib.util.find_spec(candidate)
        if spec and spec.submodule_search_locations:
            parent = Path(next(iter(spec.submodule_search_locations))).parent
            break
    if parent is None:
        return

    src_pkg = types.ModuleType("src")
    src_pkg.__path__ = [str(parent)]
    src_pkg._nnds_alias = True  # type: ignore[attr-defined]
    sys.modules["src"] = src_pkg


def _import_either(rel_path: str) -> Any:
    """Import by relative path, trying installed layout and src-layout.

    The package ships in src-layout (`pip install -e .` puts `analysis`,
    `pipeline`, ... at the top level of sys.path). The repo's internal
    modules use `from src.X import Y` everywhere. This function loads the
    module via the installed layout and ensures the `src.*` alias exists
    *before* loading, so the module's own transitive `from src.X import Y`
    statements resolve.
    """

    last_err: Exception | None = None
    for candidate in (rel_path, f"src.{rel_path}"):
        if not candidate.startswith("src."):
            _ensure_src_namespace()
        try:
            return __import__(candidate, fromlist=["*"])
        except ModuleNotFoundError as e:
            last_err = e
            continue
    raise ModuleNotFoundError(
        f"could not import {rel_path!r} from installed or src-layout path"
    ) from last_err


_SUPPORTED_DETECTORS = {"uvh-coco-fused"}


def run_video_to_pet(
    video_path,
    bev_config_path="configs/bev_config.json",
    grid_config_path="configs/GITI_grid_config.json",
    sam3_weights_path="sam3.pt",
    out_csv_path="outputs/petevents_bev.csv",
    pet_threshold=2.0,
    max_frames=None,
    detector="uvh-coco-fused",
    rtdetr_weights_path="rtdetr-l.pt",
    yolo_weights_path="data/models/yolo11n.pt",
    uvh_model_path="data/models/uvh26.pt",
    coco_person_model_path="data/models/yolo11n.pt",
    uvh_conf=0.20,
    coco_person_conf=0.20,
    imgsz=1280,
    person_suppress_overlap=0.35,
    device="auto",
    backend="auto",
    max_frame_gap=5,
    max_spatial_jump=30.0,
    prediction_tolerance=80.0,
    video_source=None,
    time_of_day_label=None,
    gate_config_path="configs/gate_config.yaml",
    show_progress: bool = True,
    backend_loader=None,
):
    if not Path(str(video_path)).exists():
        raise SystemExit(f"Video file not found: {video_path}")

    detector = str(detector).lower()
    if detector not in _SUPPORTED_DETECTORS:
        raise ValueError(
            f"Unsupported detector: {detector!r}. Supported: {sorted(_SUPPORTED_DETECTORS)}"
        )

    if detector != "uvh-coco-fused":
        raise ValueError(
            f"Unsupported detector: {detector!r}. "
            "Only 'uvh-coco-fused' is supported."
        )

    if not Path(str(coco_person_model_path)).exists():
        raise FileNotFoundError(
            f"COCO person model not found: {coco_person_model_path}"
        )

    if not Path(str(uvh_model_path)).exists():
        raise FileNotFoundError(
            f"UVH model not found: {uvh_model_path}"
        )

    loader = backend_loader or _import_either
    uvh_mod = loader(
        "analysis.grid_trajectory.uvh_coco_fused_grid_pet"
    )

    result = uvh_mod.run_uvh_coco_fused_grid_pet(
        video_path=str(video_path),
        bev_config_path=str(bev_config_path),
        grid_config_path=str(grid_config_path),
        output_csv_path=str(out_csv_path),
        pet_threshold=pet_threshold,
        max_frames=max_frames,
        uvh_model_path=str(uvh_model_path),
        coco_person_model_path=str(coco_person_model_path),
        uvh_conf=uvh_conf,
        coco_person_conf=coco_person_conf,
        person_suppress_overlap=person_suppress_overlap,
        imgsz=imgsz,
        device=device,
        backend=backend,
        max_frame_gap=max_frame_gap,
        max_spatial_jump=max_spatial_jump,
        prediction_tolerance=prediction_tolerance,
        video_source=video_source,
        time_of_day_label=time_of_day_label,
        gate_config_path=str(gate_config_path),
        show_progress=show_progress,
    )

    events = _extract_events(result)

    if not events:
        warnings.warn(
            "No PET events detected.",
            RuntimeWarning,
            stacklevel=2,
        )

    return _write_events_to_csv(events, out_csv_path)


def run_video_to_pet_fixed(
    video_path,
    bev_config_path="configs/bev_config.json",
    grid_config_path="configs/GITI_grid_config.json",
    sam3_weights_path="sam3.pt",
    out_csv_path="outputs/petevents_bev_fixed.csv",
    pet_threshold=2.0,
    max_frames=None,
):
    sam3_mod = _import_either("analysis.grid_trajectory.sam3_grid_pet")

    result = sam3_mod.run_sam3_grid_pet(
        project_root=str(Path.cwd()),
        video_rel_path=str(Path(video_path).resolve()),
        sam3_rel_path=str(Path(sam3_weights_path).resolve()),
        grid_rel_path=str(Path(grid_config_path).resolve()),
        bev_rel_path=str(Path(bev_config_path).resolve()),
        pet_threshold=pet_threshold,
        max_frames=max_frames,
    )
    events = _extract_events(result)
    return _write_events_to_csv(events, out_csv_path)


# ---------------------------------------------------------------------------
# Pipeline / CLI
# ---------------------------------------------------------------------------


def run_pipeline(args):
    """Execute the video-to-PET pipeline from parsed CLI args.

    Forwards every CLI argument that maps to a `run_video_to_pet` kwarg.
    """
    detector = str(getattr(args, "detector", "")).lower()
    if detector not in _SUPPORTED_DETECTORS:
        raise ValueError(f"Unsupported detector policy: {detector}")

    def g(name, default=None):
        return getattr(args, name, default)

    return run_video_to_pet(
        video_path=args.video,
        bev_config_path=g("bev_config") or "configs/bev_config.json",
        grid_config_path=g("grid_config") or "configs/GITI_grid_config.json",
        sam3_weights_path=g("sam3_weights", "sam3.pt"),
        out_csv_path=g("out_csv") or "outputs/petevents_bev.csv",
        pet_threshold=g("pet_threshold", 2.0),
        max_frames=g("max_frames"),
        detector=detector,
        yolo_weights_path=g("yolo_weights", "data/models/yolo11n.pt"),
        uvh_model_path=g("uvh_model", "data/models/uvh26.pt"),
        coco_person_model_path=g("coco_person_model", "data/models/yolo11n.pt"),
        rtdetr_weights_path=g("rtdetr_weights", "rtdetr-l.pt"),
        device=g("device", "auto"),
        max_frame_gap=g("max_gap", 5),
        max_spatial_jump=g("max_jump", 30.0),
        video_source=g("video_source"),
        time_of_day_label=g("time_of_day"),
        gate_config_path=g("gate_config", "configs/gate_config.yaml"),
        show_progress=not g("no_progress", False),
    )


def _validate_args(args: argparse.Namespace) -> None:
    """Validate numeric CLI constraints before expensive pipeline setup."""
    if getattr(args, "max_frames", None) is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be a positive integer.")

    if getattr(args, "pet_threshold", 2.0) < 0:
        raise ValueError("--pet-threshold must be non-negative.")

    if getattr(args, "max_gap", 5) < 0:
        raise ValueError("--max-gap must be non-negative.")

    if getattr(args, "max_jump", 30.0) < 0:
        raise ValueError("--max-jump must be non-negative.")

    if getattr(args, "samples", 0) < 0:
        raise ValueError("--samples must be non-negative.")


def run_demo():
    analyzer = CompleteTrafficAnalyzer(bev_width=100, bev_height=100)
    H = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 1.0]])
    analyzer.homography = H
    analyzer.inv_homography = np.linalg.inv(H)
    pixel_positions = np.array([[i * 10.0, 100.0] for i in range(20)], dtype=np.float32)
    frame_times = np.arange(20) / 30.0
    speed_results = analyzer.estimate_speed(pixel_positions, frame_times)
    metrics = {"mae": 0.0}
    return analyzer, speed_results, metrics


def interactive_detector(frame, model):
    results = model(frame)
    if not results:
        return []
    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    names = getattr(model, "names", {}) or {}
    detections = []
    for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls, strict=False):
        k = int(k)
        detections.append(
            {
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "conf": float(c),
                "cls": names.get(k, str(k)),
                "cls_id": k,
            }
        )
    return detections


def _version_string() -> str:
    """Compose the --version string: package version + git revision."""
    import subprocess

    try:
        from importlib.metadata import version as pkg_version

        pkg = pkg_version("nnds")
    except Exception:
        pkg = "dev"
    # If git is unavailable or the checkout has no revision info, fall back
    # to "unknown". Not a failure condition — it only affects the version string.
    rev = "unknown"
    with contextlib.suppress(Exception):
        rev = (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=2,
            ).stdout.strip()
            or "unknown"
        )
    return f"traffic_analyzer {pkg} (revision {rev})"


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for `nnds`. All flags documented and grouped."""
    epilog = (
        "examples:\n"
        "  # Main entry point (MEP)\n"
        "  nnds --video path/to/video.mp4 --detector uvh-coco-fused ...\n"
        "\n"
        "  # Optional wrapper that prepares models before dispatching to nnds\n"
        "  nnds-run --video path/to/video.mp4 ...\n"
        "\n"
        "  # Fast sanity check (no model weights required)\n"
        "  nnds --demo\n"
        "\n"
        "  # Verify environment and models for a specific video\n"
        "  nnds --self-check --video path/to/video.mp4\n"
        "\n"
        "  # Full pipeline run with UVH + COCO person detector\n"
        "  nnds --video path/to/video.mp4 --detector uvh-coco-fused \\\n"
        "       --uvh-model data/models/uvh26.pt \\\n"
        "       --coco-person-model data/models/yolo11n.pt\n"
        "\n"
        "Documentation: https://github.com/Sharqascc/nnds"
    )
    parser = argparse.ArgumentParser(
        prog="nnds",
        description=(
            "NNDS main entry point (MEP): detect vehicles and pedestrians, "
            "track them, project to bird's-eye view, and compute PET "
            "(post-encroachment time) safety events."
        ),
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    modes = parser.add_argument_group("modes (mutually exclusive)")
    modes.add_argument(
        "--demo",
        action="store_true",
        help="run a self-contained demo on a built-in clip (no weights required)",
    )
    modes.add_argument(
        "--self-check",
        action="store_true",
        help="run preflight checks (video present, models loadable, configs parseable) and exit",
    )
    modes.add_argument(
        "--dry-run",
        action="store_true",
        help="run preflight without executing the pipeline (alias of --self-check for clarity)",
    )
    modes.add_argument(
        "--version",
        action="version",
        version=_version_string(),
        help="print package version and git revision, then exit",
    )

    inp = parser.add_argument_group("input")
    inp.add_argument(
        "--video",
        type=str,
        default=None,
        metavar="PATH",
        help="path to the input video file (mp4 / avi). Required unless a mode flag is set.",
    )
    inp.add_argument(
        "--max-frames",
        dest="max_frames",
        type=int,
        default=None,
        metavar="N",
        help="stop after N frames (default: process entire video)",
    )

    det = parser.add_argument_group("detector and models")
    det.add_argument(
        "--detector",
        type=str,
        default="uvh-coco-fused",
        choices=sorted(_SUPPORTED_DETECTORS),
        help=(
            "detector backend: uvh-coco-fused (default)."
        ),
    )
    det.add_argument(
        "--uvh-model",
        dest="uvh_model",
        type=str,
        default="data/models/uvh26.pt",
        metavar="PATH",
        help="UVH-26 detector weights (used with --detector uvh-coco-fused)",
    )
    det.add_argument(
        "--coco-person-model",
        dest="coco_person_model",
        type=str,
        default="data/models/yolo11n.pt",
        metavar="PATH",
        help="COCO person-class YOLO weights (used with --detector uvh-coco-fused)",
    )
    det.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="inference device (auto picks cuda if available, else cpu)",
    )

    cfg = parser.add_argument_group("configs")
    cfg.add_argument(
        "--bev-config",
        dest="bev_config",
        type=str,
        default=None,
        metavar="PATH",
        help="bird's-eye-view calibration JSON (default: configs/bev_config.json)",
    )
    cfg.add_argument(
        "--grid-config",
        dest="grid_config",
        type=str,
        default=None,
        metavar="PATH",
        help="spatial grid config JSON (default: configs/GITI_grid_config.json)",
    )
    cfg.add_argument(
        "--gate-config",
        dest="gate_config",
        type=str,
        default="configs/gate_config.yaml",
        metavar="PATH",
        help="entry/exit gate definitions for gate counting",
    )

    trk = parser.add_argument_group("tracking")
    trk.add_argument(
        "--max-gap",
        dest="max_gap",
        type=int,
        default=5,
        metavar="FRAMES",
        help="max consecutive missing frames before a track is split (default: 5)",
    )
    trk.add_argument(
        "--max-jump",
        dest="max_jump",
        type=float,
        default=30.0,
        metavar="PIXELS",
        help="max inter-frame spatial jump when splitting tracks (default: 30)",
    )

    out = parser.add_argument_group("output and metadata")
    out.add_argument(
        "--out-csv",
        dest="out_csv",
        type=str,
        default=None,
        metavar="PATH",
        help="output PET events CSV (default: outputs/petevents_bev.csv)",
    )
    out.add_argument(
        "--out-video",
        dest="out_video",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "optional annotated MP4 that renders grid, gates, tracked boxes, "
            "and PET events over the input video"
        ),
    )
    out.add_argument(
        "--samples",
        dest="samples",
        type=int,
        default=0,
        metavar="N",
        help=(
            "optional: write N annotated sample PET-pair images showing the "
            "N lowest-PET events with bounding boxes, trajectories, grid "
            "cells, and PET values (default: 0 = off)"
        ),
    )
    out.add_argument(
        "--pet-threshold",
        dest="pet_threshold",
        type=float,
        default=2.0,
        metavar="SECONDS",
        help="max PET value (seconds) to report as a conflict (default: 2.0)",
    )
    out.add_argument(
        "--video-source",
        dest="video_source",
        type=str,
        default=None,
        metavar="NAME",
        help="site / source name recorded in each PET event (e.g. GITI, MRC)",
    )
    out.add_argument(
        "--time-of-day",
        dest="time_of_day",
        type=str,
        default=None,
        metavar="LABEL",
        help="time-of-day label recorded in each PET event (e.g. morning, evening)",
    )
    out.add_argument(
        "--no-progress",
        dest="no_progress",
        action="store_true",
        help="suppress tqdm progress bars (useful in scripts and CI logs)",
    )

    return parser


def parse_args(argv=None):
    return _build_parser().parse_args(argv)


def _render_video(args) -> None:
    """Post-pipeline: render annotated MP4 from the CSVs the run produced."""
    render_full_video = _import_either("analysis.visualization.render_full_video").render_full_video

    out_csv_path = args.out_csv or "outputs/petevents_bev.csv"
    det_csv_path = out_csv_path.replace(".csv", "_detections.csv")
    detections_csv = det_csv_path if Path(det_csv_path).exists() else None

    print(f"\nRendering video → {args.out_video}")
    try:
        summary = render_full_video(
            video_path=str(args.video),
            events_csv=out_csv_path,
            detections_csv=detections_csv,
            grid_config=args.grid_config,
            gate_config=args.gate_config,
            out_path=args.out_video,
            max_frames=args.max_frames,
        )
    except Exception as e:
        print(f"\n❌ Video render failed: {e}", file=sys.stderr)
        print(f"   CSV was written to {out_csv_path}", file=sys.stderr)
        raise SystemExit(1) from e

    print(
        f"✅ Video written → {summary['out_path']} "
        f"({summary['frames_written']} frames, "
        f"{summary['events_drawn']} events annotated)"
    )


def _render_samples(args) -> None:
    """Post-pipeline: write N sample PET-pair images for the N lowest-PET events."""
    try:
        render_pet_samples = _import_either("analysis.visualization.pet_samples").render_pet_samples
    except Exception as exc:
        print(f"\n⚠️  pet_samples unavailable: {exc}", file=sys.stderr)
        return

    out_csv_path = args.out_csv or "outputs/petevents_bev.csv"
    det_csv_path = out_csv_path.replace(".csv", "_detections.csv")
    out_dir = str(Path(out_csv_path).parent / "samples")

    if not Path(out_csv_path).exists():
        print(
            f"\n⚠️  cannot render samples: {out_csv_path} missing",
            file=sys.stderr,
        )
        return
    if not Path(det_csv_path).exists():
        print(
            f"\n⚠️  cannot render samples: {det_csv_path} missing",
            file=sys.stderr,
        )
        return

    print(f"\nRendering {args.samples} sample PET images -> {out_dir}")
    try:
        paths = render_pet_samples(
            events_csv=out_csv_path,
            detections_csv=det_csv_path,
            video_path=args.video,
            out_dir=out_dir,
            n=args.samples,
            grid_config_path=getattr(args, "grid_config", None),
        )
        if paths:
            print(f"✅ {len(paths)} samples written to {out_dir}")
        else:
            print("   (no samples produced)")
    except Exception as exc:
        print(f"\n❌ Sample rendering failed: {exc}", file=sys.stderr)
        print(f"   CSV was written to {out_csv_path}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    """Run the NNDS video-to-PET CLI.

    Args:
        argv: Optional argument list. Defaults to `sys.argv[1:]`, matching
            standard `argparse` behavior and making the CLI testable.

    Returns:
        Process exit code. Console-script entry points convert a returned
        integer into the process exit code.
    """
    logging.basicConfig(level=logging.INFO)

    # Preserve compatibility with tests and callers that replace
    # `parse_args` with a zero-argument callable.
    args = parse_args() if argv is None else parse_args(argv)
    _validate_args(args)

    if getattr(args, "self_check", False) or getattr(args, "dry_run", False):
        return _run_self_check(args)

    if getattr(args, "demo", False):
        run_demo()
        return 0

    video = getattr(args, "video", None)
    if not video:
        raise SystemExit(
            "--video is required. Try --demo for a sample run, or "
            "--self-check to verify your environment."
        )

    if not Path(str(video)).exists():
        raise SystemExit(f"Video file not found: {video}")

    # Preserve the existing early rejection: RT-DETR is parser-compatible
    # but its video pipeline is not implemented.
    if args.detector == "rtdetr":
        raise NotImplementedError(
            "RT-DETR video pipeline is not implemented. "
            "Use --detector uvh-coco-fused instead."
        )

    try:
        run_pipeline(args)
    except KeyboardInterrupt:
        logging.getLogger(__name__).warning("Interrupted by user.")
        return 130

    if getattr(args, "out_video", None):
        _render_video(args)

    if getattr(args, "samples", 0) > 0:
        _render_samples(args)

    return 0


def _run_self_check(args: argparse.Namespace) -> int:
    """Run preflight checks and return the appropriate exit code."""
    import json
    from importlib import import_module

    preflight = None
    for candidate in ("pipeline.preflight", "src.pipeline.preflight", "preflight"):
        try:
            preflight = import_module(candidate)
            break
        except ModuleNotFoundError:
            continue

    if preflight is None:
        raise SystemExit(
            "ERROR: could not locate preflight module "
            "(tried pipeline.preflight, src.pipeline.preflight, preflight)"
        )

    report = preflight.run_checks(
        video=args.video,
        uvh_model=args.uvh_model,
        coco_person_model=args.coco_person_model,
        bev_config=args.bev_config or "configs/bev_config.json",
        grid_config=args.grid_config or "configs/GITI_grid_config.json",
        gate_config=args.gate_config or "configs/gate_config.yaml",
        out_csv=args.out_csv or "outputs/petevents_bev.csv",
        device=args.device,
        detector=args.detector,
    )

    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.ok else 1


if __name__ == "__main__":
    main()
