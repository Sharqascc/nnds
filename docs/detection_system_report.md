> **DEPRECATED** — This document contains historical results. See `STATUS.md` for current results.

# Detection System Full Details Report

**Generated on:** 2026-08-25T13:08:13.654251  
**Repository:** Sharqascc/nnds  
**Branch:** cleanup/system-reorganization  

## 1. Repository Structure

```
CHANGELOG.md
CITATION.cff
CONTRIBUTING.md
DATA_LICENSE
Dockerfile
LICENSE
Makefile
PRIVACY.md
README.md
baselines/
  constant_acceleration.py
  constant_velocity.py
  kalman_filter.py
  social_force.py
configs/
  GITI_grid_config.json
  bev_config.json
  gate_config.yaml
  giti_calibration_points.json
  tracktrack_reid.yaml
  tracktrack_reid_strong.yaml
data/
  models/
    uvh26.pt
    uvh26_openvino_model/
      metadata.yaml
      uvh26.bin
      uvh26.xml
    yolo11n.pt
    yolo11n_openvino_model/
      metadata.yaml
      yolo11n.bin
      yolo11n.xml
  sample_data/
    anonymized_traffic_video_50f.mp4
    traffic_video.mp4
docker-compose.yml
docs/
  CLEANUP_SUMMARY.md
  DEBUGGING.md
  FREE_VLM_MODELS.md
  MIGRATION_GUIDE.md
  PUBLICATION_READINESS.md
  archive/
    code_dumps/
      pipeline_code_dump_2026-04-16.txt
  data_samples/
    petevents_bev_demo.csv
  detection_system_report.md
  figures/
    dependency_graph.png
    nnds_full_deps.png
  repo_full_details.md
  repository_assessment.md
environment.yml
examples/
  quickstart.py
model_cards/
  uvh26.md
  yolo11n.md
outputs/
  bev_validation_overlay.png
  compare_pet.csv
  compare_pet_detections.csv
  compare_pet_fixed.csv
  compare_pet_fixed3.csv
  compare_pet_fixed3_detections.csv
  compare_pet_fixed_detections.csv
  tracking_overlap_debug.log
pyproject.toml
requirements.txt
scripts/
  anonymize_video.py
  convert_del4_to_diffusion.py
  convert_pet_to_diffusion_csv.py
  debug_tracking_video.py
  diagnose_tracking.py
  download_models.sh
  ensure_models.py
  evaluate_ground_truth.py
  evaluate_position_ddpm.py
  evaluate_transformer_diffusion.py
  experiment_logger.py
  export_openvino.py
  grid_search_smoothing.py
  inspect_pet.py
  paired_ttest.py
  run_pipeline.py
  split_detections.py
  tracking_report.py
  tracking_report_fast.py
  traffic_analyzer_demo.py
  train_position_ddpm.py
  train_transformer_diffusion.py
  validate_all.py
  validate_bev.py
  validate_outputs.py
  validation_report.py
  visualize_pet.py
  visualize_pet_live.py
src/
  __init__.py
  analysis/
    __init__.py
    audit/
      __init__.py
      audit_config.json
    gate_counter.py
    grid_trajectory/
      __init__.py
      pet_grid.py
      sam3_grid_pet.py
      spatial_grid.py
      uvh_coco_fused_grid_pet.py
      yolo_cpu_grid_pet.py
    logging/
      __init__.py
      reproducibility_audit.py
    pet_conflict_checker.py
    pet_diffusion_analysis.py
    pet_summary.py
    research_run.py
    safety_eval_diffusion.py
    safety_eval_diffusion_notebook.py
    ssm/
      __init__.py
      ssm_verification.py
      uncertainty_quantifier.py
    verification/
      __init__.py
      statistical_testing.py
    visualization/
      __init__.py
      industry_standard_viz.py
      pet_diffusion_plots.py
      pet_event_plots.py
      video_overlays.py
  bev/
    __init__.py
    bev_mapper.py
    calibration/
      MANIFEST.json
      PROVENANCE.md
      README.md
      REPRODUCIBILITY.md
      __init__.py
      grid_validation_calibration.py
      monte_carlo_calibration_benchmark.py
      monte_carlo_calibration_notes.md
    giti_bev_calib.py
  core/
    __init__.py
    types.py
    validation.py
  diffusion/
    __init__.py
    complete_ddpm.py
    traffic_diffusion/
      __init__.py
      data/
      evaluate_fixed.py
      model_and_sampler.py
      mypy.ini
      sampling_utils.py
      split_dataset.py
      train_trajectory_diffusion.py
      training_utils.py
      trajectory_diffusion.py
      transformer_diffusion.py
    traj_diffusion_normalized.py
  pipeline/
    __init__.py
    custom_tracker.py
    reid_encoder.py
    rt_detr_detector.py
    traffic_analyzer.py
  utils/
    __init__.py
    debug_helpers.py
    interactive.py
    seed.py
  vlm/
    __init__.py
    analyzer.py
    config.py
    gate_validator.py
    requirements.txt
    test_free_models.py
    utils/
      __init__.py
      image_utils.py
      visualization.py
    vlm_enhanced_pipeline.py
tests/
  __init__.py
  conftest.py
  fixtures/
    ground_truth_sample.csv
    sample_detections.csv
    sample_pet.csv
    sample_split_detections.csv
  test_baselines_extra.py
  test_baselines_seed.py
  test_bev_validation.py
  test_configs_smoke.py
  test_diffusion_smoke.py
  test_imports_smoke.py
  test_modules_smoke.py
  test_new_scripts.py
  test_paired_ttest.py
  test_pet_conflict.py
  test_pet_logic.py
  test_pet_output_schema.py
  test_repo_smoke.py
  test_research_run_smoke.py
  test_rtdetr_stub.py
  test_smoke.py
  test_speed_estimation.py
  test_traffic_analyzer_cli.py
  test_traffic_analyzer_demo_smoke.py
  test_validation.py
```

## 2. Detection-Related Files

| File | Lines |
|------|-------|
| src/analysis/grid_trajectory/uvh_coco_fused_grid_pet.py | 727 |
| src/analysis/grid_trajectory/yolo_cpu_grid_pet.py | 273 |
| src/analysis/grid_trajectory/sam3_grid_pet.py | 451 |
| src/pipeline/rt_detr_detector.py | 35 |
| src/pipeline/custom_tracker.py | 293 |
| src/pipeline/traffic_analyzer.py | 1286 |

## 3. Main UVH-COCO Fused Detection Pipeline

### File: `uvh_coco_fused_grid_pet.py`

```python
from __future__ import annotations
import matplotlib.pyplot as plt
from src.utils.interactive import show_image, ask_user
from src.analysis.grid_trajectory.spatial_grid import SpatialGrid
from src.bev.bev_mapper import BEVMapper
from tqdm import tqdm
import sys
import json


from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Tuple

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import torch
from src.pipeline.custom_tracker import CustomTracker, Detection
from src.pipeline.reid_encoder import ReIDEncoder

def _compute_histogram(frame, x1, y1, x2, y2):
    """Compute normalized HSV histogram for a crop."""
    try:
        x1i, y1i = max(0, int(x1)), max(0, int(y1))
        x2i, y2i = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
        crop = frame[y1i:y2i, x1i:x2i]
        if crop.size == 0:
            return None
        # Convert to HSV
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # 2D histogram over H and S
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()
    except Exception:
        return None



UVH_DISPLAY_MAP = {
    "Three-wheeler": "auto",
    "Two-wheeler": "bike",
    "Hatchback": "car",
    "Sedan": "car",
    "SUV": "car",
    "MUV": "car",
    "Van": "car",
    "Truck": "truck",
    "LCV": "truck",
    "Bus": "bus",
    "Mini-bus": "bus",
    "tempo-traveller": "bus",
}

CLASS_NAME_TO_ID = {
    "pedestrian": 0,
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "bike": 3,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7,
    "auto": 8,
}


@dataclass
class TrackPoint:
    frame: int
    x: float
    y: float
    cls_id: int
    cls_name: str
    conf: float


def _segment_intersection(p1, p2, q1, q2):
    p = np.array(p1, dtype=float)
    r = np.array(p2, dtype=float) - p
    q = np.array(q1, dtype=float)
    s = np.array(q2, dtype=float) - q

    rxs = np.cross(r, s)
    q_p = q - p
    qpxr = np.cross(q_p, r)

    if abs(rxs) < 1e-9 and abs(qpxr) < 1e-9:
        return None
    if abs(rxs) < 1e-9 and abs(qpxr) >= 1e-9:
        return None

    t = np.cross(q_p, s) / rxs
    u = np.cross(q_p, r) / rxs

    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        inter = p + t * r
        return float(inter[0]), float(inter[1])
    return None


def _point_in_square(px, py, cx, cy, half_size):
    return (cx - half_size) <= px <= (cx + half_size) and (cy - half_size) <= py <= (cy + half_size)


def _entry_exit_frames(points: List[TrackPoint], cx: float, cy: float, half_size: float):
    inside_frames = [pt.frame for pt in points if _point_in_square(pt.x, pt.y, cx, cy, half_size)]
    if not inside_frames:
        return None
    return min(inside_frames), max(inside_frames)


def _segment_bbox(p1, p2):
    return (min(p1[0], p2[0]), min(p1[1], p2[1]), max(p1[0], p2[0]), max(p1[1], p2[1]))

def _bbox_overlap(box1, box2, pad=0.0):
    return not (box1[2] < box2[0] - pad or box2[2] < box1[0] - pad or
                box1[3] < box2[1] - pad or box2[3] < box1[1] - pad)


def _compute_pet_from_windows(a_entry, a_exit, b_entry, b_exit, fps):
    """
    Pure function: compute Post‑Encroachment Time from two entry/exit windows.

    Returns:
        (pet, first_id, second_id, frame_ref) if a valid PET can be computed,
        otherwise None.
    """
    if a_exit <= b_entry:
        pet = (b_entry - a_exit) / fps
        first_id, second_id = "a", "b"
        frame_ref = b_entry
    elif b_exit <= a_entry:
        pet = (a_entry - b_exit) / fps
        first_id, second_id = "b", "a"
        frame_ref = a_entry
    else:
        return None
    return pet, first_id, second_id, frame_ref


def _pair_conflict_point(track_a: List[TrackPoint], track_b: List[TrackPoint]):
    for i in range(len(track_a) - 1):
        p1 = (track_a[i].x, track_a[i].y)
        p2 = (track_a[i + 1].x, track_a[i + 1].y)
        bbox_a = _segment_bbox(p1, p2)
        for j in range(len(track_b) - 1):
            q1 = (track_b[j].x, track_b[j].y)
            q2 = (track_b[j + 1].x, track_b[j + 1].y)
            bbox_b = _segment_bbox(q1, q2)
            if not _bbox_overlap(bbox_a, bbox_b):
                continue
            inter = _segment_intersection(p1, p2, q1, q2)
            if inter is not None:
                return inter
    return None


def _box_intersection(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float((x2 - x1) * (y2 - y1))


def _box_area(box: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _overlap_over_person(person_box: Tuple[float, float, float, float], other_box: Tuple[float, float, float, float]) -> float:
    inter = _box_intersection(person_box, other_box)
    area = _box_area(person_box)
    return 0.0 if area <= 0 else inter / area


def run_uvh_coco_fused_grid_pet(
    video_path: str,
    bev_config_path: str,
    grid_config_path: str,
    uvh_model_path: str,
    coco_person_model_path: str,
    output_csv_path: str,
    pet_threshold: float = 2.0,
    max_frames: int | None = None,
    imgsz: int = 1280,
    uvh_conf: float = 0.20,
    coco_person_conf: float = 0.20,
    person_suppress_overlap: float = 0.35,
    show_progress: bool = True, interactive: bool = False,
    device: str = "auto",
    backend: str = "auto",
) -> Dict[str, Any]:
    video_path = str(Path(video_path).resolve())
    uvh_model_path = str(Path(uvh_model_path).resolve())
    coco_person_model_path = str(Path(coco_person_model_path).resolve())
    output_csv_path = str(Path(output_csv_path).resolve())

    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"


... (truncated, full file has 727 lines)
```

## 4. Traffic Analyzer (Entry Point)

```python
def run_video_to_pet(
    video_path: Path | str,
    bev_config_path: Path | str = "configs/bev_config.json",
    grid_config_path: Path | str = "configs/GITI_grid_config.json",
    sam3_weights_path: Path | str = "sam3.pt",
    out_csv_path: Path | str = "outputs/petevents_bev.csv",
    pet_threshold: float = 2.0,
    max_frames: int | None = None,
    show_progress: bool = True,
    detector: str = "uvh-coco-fused",
    rtdetr_weights_path: Path | str = "rtdetr-l.pt",
    yolo_weights_path: Path | str = "data/models/yolo11n.pt",
    uvh_model_path: Path | str = "data/models/uvh26.pt",
    coco_person_model_path: Path | str = "data/models/yolo11n.pt",
    uvh_conf: float = 0.20,
    coco_person_conf: float = 0.20,
    imgsz: int = 1280,
    person_suppress_overlap: float = 0.35,
    device: str = "auto",
    backend: str = "auto",
) -> pd.DataFrame:
    """Video → detections → grid → BEV → PET events CSV (SAM3 or RT-DETR).

    NOTE: RT-DETR path is currently a placeholder and must be implemented.
    """

    video_path = Path(video_path)
    bev_config_path = Path(bev_config_path)
    grid_config_path = Path(grid_config_path)
    sam3_weights_path = Path(sam3_weights_path)
    rtdetr_weights_path = Path(rtdetr_weights_path)
    yolo_weights_path = Path(yolo_weights_path)
    uvh_model_path = Path(uvh_model_path)
    coco_person_model_path = Path(coco_person_model_path)
    out_csv_path = Path(out_csv_path)

    # Validate inputs early with clear messages
    for path, name in [
        (video_path, "Video"),
        (bev_config_path, "BEV config"),
        (grid_config_path, "Grid config"),
    ]:
        if not path.exists():
            raise SystemExit(f"Video file not found: {path}")

    if detector == "sam3":
        # SAM3 path: validate SAM3 weights and run existing pipeline
        if not sam3_weights_path.exists():
            sam3_weights_path = None

        try:
            from src.analysis.grid_trajectory.sam3_grid_pet import run_sam3_grid_pet
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Missing dependency for video pipeline. Install required packages "
                "for SAM3/Ultralytics before running video mode "
                "(e.g., `pip install ultralytics supervision`)."
            ) from exc

        project_root = str(Path.cwd())
        result = run_sam3_grid_pet(
            project_root=project_root,
            video_rel_path=str(video_path),
            sam3_rel_path=str(sam3_weights_path),
            grid_rel_path=str(grid_config_path),
            bev_rel_path=str(bev_config_path),
            output_name="sam3_grid_pet_run",
            conf=0.25,
            pet_threshold=pet_threshold,
            max_frames=max_frames,
            show_progress=show_progress,
        )
        pet_events = result.pet_events if hasattr(result, "pet_events") else []

    elif detector == "yolo-cpu":
        if not yolo_weights_path.exists():
            raise FileNotFoundError(f"YOLO weights not found: {yolo_weights_path}")

        try:
            from src.analysis.grid_trajectory.yolo_cpu_grid_pet import (
                run_yolo_cpu_grid_pet,
            )
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Missing dependency for YOLO CPU pipeline."
            ) from exc

        result = run_yolo_cpu_grid_pet(
            video_path=str(video_path),
            weights_path=str(yolo_weights_path),
            output_csv_path=str(out_csv_path),
            max_frames=max_frames,
            imgsz=480,
            conf=0.25,
        )
        pet_events = (
            result["pet_events"]
            if isinstance(result, dict) and "pet_events" in result
            else []
        )

    elif detector == "uvh-coco-fused":
        if not uvh_model_path.exists():
            raise FileNotFoundError(f"UVH model not found: {uvh_model_path}")
        if not coco_person_model_path.exists():
            raise FileNotFoundError(
                f"COCO person model not found: {coco_person_model_path}"
            )

        try:
            from src.analysis.grid_trajectory.uvh_coco_fused_grid_pet import (
                run_uvh_coco_fused_grid_pet,
            )
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Missing fused detector backend. Expected "
                "grid_trajectory.uvh_coco_fused_grid_pet.run_uvh_coco_fused_grid_pet"
            ) from exc

        result = run_uvh_coco_fused_grid_pet(
            video_path=video_path,
            bev_config_path=str(bev_config_path),
            grid_config_path=str(grid_config_path),
            uvh_model_path=str(uvh_model_path),
            coco_person_model_path=str(coco_person_model_path),
            output_csv_path=str(out_csv_path),
            pet_threshold=pet_threshold,
            max_frames=max_frames,
            imgsz=imgsz,
            uvh_conf=uvh_conf,
            coco_person_conf=coco_person_conf,
            person_suppress_overlap=person_suppress_overlap,
            show_progress=show_progress,
            device=device,
            backend=backend,
        )
        pet_events = (
            result["pet_events"]
            if isinstance(result, dict) and "pet_events" in result
            else []
        )

    else:
        # RT-DETR path declared but not implemented on this branch
        if not rtdetr_weights_path.exists():
            raise FileNotFoundError(f"RT-DETR weights not found: {rtdetr_weights_path}")

        raise NotImplementedError(
            "RT-DETR video pipeline is not implemented in this branch. "
            "Missing module: grid_trajectory.rtdetr_grid_pet.run_rtdetr_grid_pet. "
            "Use detector='sam3' or add grid_trajectory/rtdetr_grid_pet.py "
            "with a compatible run_rtdetr_grid_pet(...) implementation."
        )

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Handle empty results robustly
    if not pet_events:
        warnings.warn(f"No PET events detected in {video_path}", RuntimeWarning)
        empty_df = pd.DataFrame(
            columns=[
                "event_id",
                "pet",
                "pet_time_based",
                "frame",
                "track_a",
                "track_b",
                "orig_track_a",
                "seg_a",
                "orig_track_b",
                "seg_b",
                "conflict_type",
                "grid_cell",
                "track_a_entry_frame",
                "track_a_exit_frame",
                "track_a_exit_time_sec",
                "track_b_entry_frame",
                "track_b_entry_time_sec",
                "track_b_exit_frame",
                "world_traj_i",
                "world_traj_j",
                "traj_a_json",
                "traj_b_json",
            ]
        )
        empty_df.to_csv(out_csv_path, index=False)
        logger.info("⚠️  No PET events. Wrote empty CSV to %s", out_csv_path)
        return empty_df

    rows: list[dict[str, Any]] = []
    for idx, e in enumerate(pet_events):

        def _get(keys, default=None):
            if isinstance(e, dict):
                for k in keys:
                    if k in e and e[k] is not None:
                        return e[k]
            else:
                for k in keys:
                    if hasattr(e, k) and getattr(e, k) is not None:
                        return getattr(e, k)
            return default

        # Extract track IDs (check direct integer attributes or parse 'track_17' strings)
        def _parse_track_id(keys):
            val = _get(keys)
            if val is None or val == -1:
                return -1
            if isinstance(val, (int, float)) and not (
                isinstance(val, float) and np.isnan(val)
            ):
                return int(val)
            m = re.search(r"\d+", str(val))
            return int(m.group()) if m else -1

        track_a_val = _parse_track_id(
            ["track_a", "obj_i", "track_i", "traj_i_id", "world_traj_i"]
        )
        track_b_val = _parse_track_id(
            ["track_b", "obj_j", "track_j", "traj_j_id", "world_traj_j"]
        )
        frame_val = _get(
            ["frame", "conflict_frame", "start_frame", "frame_idx", "t_conflict"]
        )
        pet_val = _get(["PET", "pet"], float("inf"))
        conflict_type_val = _get(["conflict_type", "cell_id"], "UNKNOWN")
        grid_cell_val = _get(["grid_cell", "cell_id"], "UNKNOWN")
        entry_a_val = _get(["track_a_entry_frame"], -1)
        exit_a_val = _get(["track_a_exit_frame"], -1)
        entry_b_val = _get(["track_b_entry_frame"], -1)
        exit_b_val = _get(["track_b_exit_frame"], -1)
        world_traj_i_val = _get(["world_traj_i", "traj_i"])
        world_traj_j_val = _get(["world_traj_j", "traj_j"])
        traj_a_json_val = _get(["traj_a_json"], "[]")
        traj_b_json_val = _get(["traj_b_json"], "[]")

        rows.append(
            {
                "event_id": idx,
                "pet": pet_val,
                "pet_time_based": _get(["pet_time_based"], None),
                "frame": frame_val,
                "track_a": track_a_val,
                "track_b": track_b_val,
                "orig_track_a": _get(["orig_track_a"], -1),
                "seg_a": _get(["seg_a"], -1),
                "orig_track_b": _get(["orig_track_b"], -1),
                "seg_b": _get(["seg_b"], -1),
                "conflict_type": conflict_type_val,
                "grid_cell": grid_cell_val,
                "track_a_entry_frame": entry_a_val,
                "track_a_exit_frame": exit_a_val,
                "track_a_exit_time_sec": _get(["track_a_exit_time_sec"], None),
                "track_b_entry_frame": entry_b_val,
                "track_b_entry_time_sec": _get(["track_b_entry_time_sec"], None),
                "track_b_exit_frame": exit_b_val,
                "world_traj_i": world_traj_i_val,
                "world_traj_j": world_traj_j_val,
                "traj_a_json": traj_a_json_val,
                "traj_b_json": traj_b_json_val,
            }
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "event_id",
            "pet",
            "pet_time_based",
            "frame",
            "track_a",
            "track_b",
            "orig_track_a",
            "seg_a",
            "orig_track_b",
            "seg_b",
            "conflict_type",
            "grid_cell",
            "track_a_entry_frame",
            "track_a_exit_frame",
            "track_a_exit_time_sec",
            "track_b_entry_frame",
            "track_b_entry_time_sec",
            "track_b_exit_frame",
            "world_traj_i",
            "world_traj_j",
            "traj_a_json",
            "traj_b_json",
        ],
    )
    df.to_csv(out_csv_path, index=False)
    logger.info("✅ Saved %d PET events to %s", len(df), out_csv_path)
    return df



```

## 5. Tracker Configuration

_TrackTrack config not found in repo; using Ultralytics default._

## 6. Model Cards

### uvh26.md
# UVH-26 Model Card

- **Model:** UVH-26-MV-YOLOv11-S
- **Source:** Hugging Face iisc-aim/UVH-26
- **Task:** Multi-class vehicle detection (pedestrian, car, bike, bus, truck, auto, etc.)
- **Training data:** Proprietary Indian traffic dataset (not included)
- **Input size:** 640x640 typical
- **Framework:** Ultralytics YOLO (PyTorch)
- **Intended use:** Research in traffic conflict analysis
- **Limitations:** Trained on Indian traffic scenes; performance may vary in other regions.

### yolo11n.md
# YOLO11n Model Card

- **Model:** YOLO11n
- **Source:** Ultralytics official release
- **Task:** COCO object detection (person fallback)
- **Training data:** COCO dataset
- **Input size:** 640x640
- **Framework:** Ultralytics YOLO (PyTorch)
- **Intended use:** Person detection to supplement UVH-26
- **Limitations:** General COCO classes; not traffic-specific.


## 7. Detection Validation & Evaluation

### scripts/validate_outputs.py
```python
#!/usr/bin/env python3
"""
Automated validation of NNDS pipeline outputs.

Checks detections, tracking stability (on split tracks), and PET consistency.

Usage:
    python scripts/validate_outputs.py \
        --detections outputs/petevents_bev_final_detections.csv \
        --detections-split outputs/petevents_bev_final_split_detections.csv \
        --pet outputs/petevents_bev_final.csv
"""
import argparse
import json
import sys
import pandas as pd
import numpy as np
from pathlib import Path

DETECTION_COLUMNS = [
    "frame", "track_id", "class_id", "class_name",
    "conf", "x1", "y1", "x2", "y2", "cx", "cy", "source",
]
PET_COLUMNS = [
    "event_id", "pet", "pet_time_based", "frame", "track_a", "track_b",
    "orig_track_a", "seg_a", "orig_track_b", "seg_b",
    "conflict_type", "grid_cell",
    "track_a_entry_frame", "track_a_exit_frame", "track_a_exit_time_sec",
    "track_b_entry_frame", "track_b_entry_time_sec", "track_b_exit_frame",
    "world_traj_i", "world_traj_j", "traj_a_json", "traj_b_json",
]
ALLOWED_CLASSES = {"pedestrian", "person", "bicycle", "car", "bike", "motorcycle", "bus", "truck", "auto"}

def load_csv(path, required_columns, context):
    path = Path(path)
    if not path.exists():
        print(f"❌ {context}: file not found: {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    missing = set(required_columns) - set(df.columns)
    if missing:
        print(f"❌ {context}: missing columns: {sorted(missing)}")
        sys.exit(1)
    if df.empty:
        print(f"⚠️ {context}: empty DataFrame")
        return df, True
    return df, False

def validate_detections(det_df):
    problems = []
    bad_boxes = det_df[(det_df["x1"] >= det_df["x2"]) | (det_df["y1"] >= det_df["y2"])]
    if not bad_boxes.empty:
        problems.append(f"Detections with invalid bounding boxes: {len(bad_boxes)}")
    bad_conf = det_df[(det_df["conf"] < 0) | (det_df["conf"] > 1)]
    if not bad_conf.empty:
        problems.append(f"Detections with conf out of [0,
... (truncated)
```

### scripts/evaluate_ground_truth.py
```python
#!/usr/bin/env python3
"""
Evaluate detections against ground truth annotations.

Expected ground truth format: CSV with columns:
    frame, x1, y1, x2, y2, class_name

Detection CSV from pipeline has additional columns, but we only need
frame and bounding boxes for IoU-based matching.

Usage:
    python scripts/evaluate_ground_truth.py \
        --detections outputs/petevents_bev_detections.csv \
        --ground-truth path/to/gt.csv \
        --iou-threshold 0.5
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    det = pd.read_csv(args.detections)
    gt = pd.read_csv(args.ground_truth)

    # Filter detections to frames present in GT
    det_frames = set(det['frame'].unique())
    gt_frames = set(gt['frame'].unique())
    common_frames = det_frames & gt_frames
    print(f"Frames with GT: {len(gt_frames)}, frames with detections: {len(det_frames)}, common: {len(common_frames)}")

    matched = 0
    total_gt = 0
    total_det = 0

    for frame in sorted(common_frames):
        gt_frame = gt[gt['frame'] == frame]
        det_frame = det[det['frame'] == frame]
        total_gt += len(gt_frame)
        total_det += len(det_frame)

        matched_gt = set()
        matched_det = set()
        for gi, g in gt_fr
... (truncated)
```

## 8. Detection Output Schema

Detection CSV columns: frame, track_id, class_id, class_name, conf, x1, y1, x2, y2, cx, cy, source

Sample rows:
```
 frame  track_id  class_id class_name     conf          x1         y1          x2         y2          cx         cy source
     0         1         3       bike 0.842660 1011.086121 157.521561 1053.000366 210.838043 1032.043243 184.179802  uvh26
     0         2         8       auto 0.839481 1022.800415 284.136078 1091.758301 356.367004 1057.279358 320.251541  uvh26
     0         3         3       bike 0.807684  743.243286 283.600769  807.784729 364.736084  775.514008 324.168427  uvh26
```