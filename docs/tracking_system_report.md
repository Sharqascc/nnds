> **DEPRECATED** — This document contains historical results. See `STATUS.md` for current results.

# Tracking System Full Details Report

**Generated on:** 2026-08-25T13:14:38.194856  
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

## 2. Tracking-Related Files

| File | Lines |
|------|-------|
| src/pipeline/custom_tracker.py | 293 |
| src/pipeline/reid_encoder.py | 50 |
| configs/tracktrack_reid.yaml | 20 |
| configs/tracktrack_reid_strong.yaml | 20 |
| scripts/diagnose_tracking.py | 102 |
| scripts/tracking_report.py | 206 |
| scripts/tracking_report_fast.py | 128 |
| scripts/debug_tracking_video.py | 99 |

## 3. Custom Tracker Implementation

### File: `custom_tracker.py`

```python
"""
Custom motion-based multi-object tracker with appearance disambiguation.

Uses:
  - Kalman filters for motion prediction
  - Hungarian matching (IoU first, then predicted-center + appearance)
  - HSV histograms and deep ReID embeddings for occlusion handling
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Detection:
    frame: int
    x1: float
    y1: float
    x2: float
    y2: float
    cx: float
    cy: float
    cls_id: int
    cls_name: str
    conf: float
    source: str
    hist: np.ndarray | None = None
    embedding: np.ndarray | None = None


class KalmanTrack:
    """Kalman filter state for a single object."""

    def __init__(self, det: Detection, track_id: int):
        self.id = track_id
        self.cls_id = det.cls_id
        self.cls_name = det.cls_name
        self.source = det.source
        self.age = 0
        self.time_since_update = 0
        self.hist = det.hist if det.hist is not None else None
        self.embedding = det.embedding if det.embedding is not None else None

        w = max(det.x2 - det.x1, 1.0)
        h = max(det.y2 - det.y1, 1.0)
        self.kf = cv2.KalmanFilter(6, 4)
        self.kf.transitionMatrix = np.array(
            [
                [1, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 1],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        self.kf.measurementMatrix = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
            ],
            dtype=np.float32,
        )
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.1
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 10.0
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)
        self.kf.statePost = np.array(
            [det.cx, det.cy, w, h, 0, 0], dtype=np.float32
        ).reshape(-1, 1)

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, det: Detection):
        measurement = np.array(
            [det.cx, det.cy, max(det.x2 - det.x1, 1), max(det.y2 - det.y1, 1)],
            dtype=np.float32,
        ).reshape(-1, 1)
        self.kf.correct(measurement)
        self.time_since_update = 0
        if det.hist is not None:
            if self.hist is None:
                self.hist = det.hist
            else:
                self.hist = 0.8 * self.hist + 0.2 * det.hist
        if det.embedding is not None:
            if self.embedding is None:
                self.embedding = det.embedding
            else:
                self.embedding = 0.8 * self.embedding + 0.2 * det.embedding

    @property
    def center(self):
        s = np.asarray(self.kf.statePost).reshape(-1)
        return s[0], s[1]

    @property
    def box(self):
        s = np.asarray(self.kf.statePost).reshape(-1)
        w = max(s[2], 1)
        h = max(s[3], 1)
        return (s[0] - w / 2, s[1] - h / 2, s[0] + w / 2, s[1] + h / 2)


class CustomTracker:
    """Kalman + Hungarian tracker with appearance matching."""

    def __init__(
        self,
        max_age=60,
        min_hits=1,
        iou_threshold=0.2,
        log_overlaps=False,
        overlap_log_path="outputs/tracking_overlap_debug.log",
        reid_encoder=None,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.log_overlaps = log_overlaps
        self.reid_encoder = reid_encoder
        if log_overlaps and overlap_log_path:
            Path(overlap_log_path).parent.mkdir(parents=True, exist_ok=True)
            self.log_handle = open(overlap_log_path, "w", encoding="utf-8")
            self.log_handle.write(
                "frame,track_a,track_b,iou,pred_cx_a,pred_cy_a,pred_cx_b,pred_cy_b,box_a,box_b\n"
            )
        else:
            self.log_handle = None
        self.next_id = 1
        self.tracks = {}

    def _iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
        area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def _cosine_similarity(self, a, b):
        if a is None or b is None:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))

    def _appearance_cost(self, track_hist, det_hist, track_emb, det_emb):
        """Return a cost (0=perfect similarity) between track and detection appearance."""
        # Use deep embedding if both available
        if track_emb is not None and det_emb is not None:
            sim = self._cosine_similarity(track_emb, det_emb)
            return (1.0 - sim) * 100.0  # scale cost
        # Fallback to HSV histogram if available
        if track_hist is not None and det_hist is not None:
            return (
                cv2.compareHist(
                    track_hist.astype(np.float32),
                    det_hist.astype(np.float32),
                    cv2.HISTCMP_BHATTACHARYYA,
                )
                * 50.0
            )
        return 0.0

    def update(self, detections, frame_img=None, frame=None):
        """Update tracks with new detections; returns dict det_index -> track_id."""
        for t in self.tracks.values():
            t.predict()

        if self.log_overlaps and self.log_handle is not None:
            frame_now = (
                frame
                if frame is not None
                else (detections[0].frame if detections else -1)
            )
            track_ids = list(self.tracks.keys())
            pred_boxes = {tid: self.tracks[tid].box for tid in track_ids}
            centers = {tid: self.tracks[tid].center for tid in track_ids}
            for i in range(len(track_ids)):
                for j in range(i + 1, len(track_ids)):
                    tid_a = track_ids[i]
                    tid_b = track_ids[j]
                    iou = self._iou(pred_boxes[tid_a], pred_boxes[tid_b])
                    if iou > 0.5:
                        cx_a, cy_a = centers[tid_a]
                        cx_b, cy_b = centers[tid_b]
                        box_a = ",".join(f"{v:.1f}" for v in pred_boxes[tid_a])
                        box_b = ",".join(f"{v:.1f}" for v in pred_boxes[tid_b])
                        self.log_handle.write(
                            f"{frame_now},{tid_a},{tid_b},{iou:.3f},{cx_a:.1f},{cy_a:.1f},{cx_b:.1f},{cy_b:.1f},{box_a},{box_b}\n"
                        )

        if not detections:
            return {}

        active_ids = list(self.tracks.keys())

        # Stage 1: IoU matching with predicted boxes
        cost = np.zeros((len(active_ids), len(detections)), dtype=np.float32)
        for i, tid in enumerate(active_ids):
            pred_box = self.tracks[tid].box
            for j, det in enumerate(detections):
                det_box = (det.x1, det.y1, det.x2, det.y2)
                iou = self._iou(pred_box, det_box)
                cost[i, j] = 1.0 - iou

        row_ind, col_ind = linear_sum_assignment(cost)
        matched = {}
        matched_track_ids = set()
        unmatched_dets = set(range(len(detections)))

        for i, j in zip(row_ind, col_ind):
            if cost[i, j] < 1.0 - self.iou_threshold:  # IoU > threshold
                tid = active_ids[i]
                self.tracks[tid].update(detections[j])
                matched[j] = tid
                matched_track_ids.add(tid)
                unm
... (truncated)
```

## 4. ReID Encoder

### File: `reid_encoder.py`

```python
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision import models


class ReIDEncoder:
    """Lightweight appearance encoder using MobileNetV3-Small pretrained on ImageNet."""

    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        self.model = models.mobilenet_v3_small(weights=weights)
        self.model.classifier = torch.nn.Identity()  # remove classification head
        self.model.eval().to(self.device)
        self.transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((128, 128)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def encode_crop(self, frame_bgr, x1, y1, x2, y2):
        """Return normalized 1D numpy embedding for an image crop."""
        try:
            x1i, y1i = max(0, int(x1)), max(0, int(y1))
            x2i, y2i = (
                min(frame_bgr.shape[1], int(x2)),
                min(frame_bgr.shape[0], int(y2)),
            )
            if x2i <= x1i or y2i <= y1i:
                return None
            crop = frame_bgr[y1i:y2i, x1i:x2i]
            if crop.size == 0:
                return None
            # Convert BGR to RGB
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = self.model(tensor)
            emb = emb.cpu().numpy().reshape(-1)
            norm = np.linalg.norm(emb)
            if norm < 1e-6:
                return None
            return emb / norm
        except Exception:
            return None

```

## 5. Tracker Configurations

### tracktrack_reid.yaml

```yaml

tracker_type: tracktrack
track_high_thresh: 0.6
track_low_thresh: 0.25
new_track_thresh: 0.7
track_buffer: 30
match_thresh: 0.7
lost_match_thr: 0.0
iou_weight: 0.5
reid_weight: 0.5
conf_weight: 0.1
angle_weight: 0.05
penalty_p: 0.2
penalty_q: 0.4
reduce_step: 0.05
tai_thr: 0.55
min_track_len: 3
gmc_method: sparseOptFlow
with_reid: True
model: auto

```

### tracktrack_reid_strong.yaml

```yaml

tracker_type: tracktrack
track_high_thresh: 0.6
track_low_thresh: 0.25
new_track_thresh: 0.7
track_buffer: 60
match_thresh: 0.7
lost_match_thr: 0.0
iou_weight: 0.05
reid_weight: 0.9
conf_weight: 0.05
angle_weight: 0.0
penalty_p: 0.2
penalty_q: 0.4
reduce_step: 0.05
tai_thr: 0.55
min_track_len: 3
gmc_method: sparseOptFlow
with_reid: True
model: auto

```

## 6. Tracking Evaluation & Diagnostic Scripts

### scripts/tracking_report.py

```python
#!/usr/bin/env python3
"""
Automatic tracking evaluation & report.

Reads a detections CSV with columns:
  frame, track_id, class_name, cx, cy, x1, y1, x2, y2, source

Produces a structured text report with:
  - overall statistics
  - per-track metrics (duration, detections, jumps, gaps)
  - overlapping track pairs
  - candidate ID switches

Usage:
  python scripts/tracking_report.py --csv outputs/petevents_bev_300_custom_detections.csv
  python scripts/tracking_report.py --csv outputs/... --output outputs/tracking_report.txt
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Detection CSV path")
    parser.add_argument("--output", default=None, help="Output report path (optional)")
    parser.add_argument(
        "--max-gap", type=int, default=10, help="Max frame gap for ID switch candidate"
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=50.0,
        help="Max center distance for ID switch candidate",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"frame", "track_id", "class_name", "cx", "cy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df["frame"] = pd.to_numeric(df["frame"])
    df["track_id"] = pd.to_numeric(df["track_id"])
    df["cx"] = pd.to_numeric(df["cx"])
    df["cy"] = pd.to_numeric(df["cy"])

    lines = []
    lines.append("=" * 80)
    lines.append("TRACKING EVALUATION REPORT")
    lines.append(f"Input: {csv_path}")
    lines.append("=" * 80)
    lines.append(f"Total frames: {df['frame'].nunique()}")
    lines.append(f"Total detections: {len(df)}")
    lines.append(f"Unique tracks: {df['track_id'].nunique()}")
    lines.append("")

    track_stats = {}
    for tid, grp in df.groupby("track_id"):
        grp = grp.sort_values("frame")
        frames = grp["frame"].values
        cx = grp["cx"].values
        cy = grp["cy"].values

        if len(grp) >= 2:
            dx = np.diff(cx)
            dy = np.diff(cy)
            jumps = np.sqrt(dx * dx + dy * dy)
            max_jump = float(jumps.max())
            avg_jump = float(jumps.mean())
            gaps = np.diff(frames)
            max_gap = int(gaps.max()) if len(gaps) > 0 else 0
        else:
            max_jump = 0.0
            avg_jump = 0.0
            max_gap = 0

        cls_counts = grp["class_name"].value_counts().to_dict()
        main_cls = max(cls_counts, key=cls_counts.get) if cls_counts else "unknown"

        track_stats[tid] = {
            "start": int(frames.min()),
            "end": int(frames.max()),
            "num_det": len(grp),
            "main_class": main_cls,

... (truncated)
```

### scripts/tracking_report_fast.py

```python
#!/usr/bin/env python3
"""Fast tracking report for large datasets."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-gap", type=int, default=10)
    parser.add_argument("--max-distance", type=float, default=50.0)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df["frame"] = pd.to_numeric(df["frame"])
    df["track_id"] = pd.to_numeric(df["track_id"])
    df["cx"] = pd.to_numeric(df["cx"])
    df["cy"] = pd.to_numeric(df["cy"])

    # Precompute per-track summary once
    track_info = {}
    for tid, grp in tqdm(
        df.groupby("track_id"), desc="Summarising tracks", unit="track"
    ):
        grp = grp.sort_values("frame")
        frames = grp["frame"].values
        cx = grp["cx"].values
        cy = grp["cy"].values
        jumps = (
            np.sqrt(np.diff(cx) ** 2 + np.diff(cy) ** 2)
            if len(grp) > 1
            else np.array([])
        )
        max_gap = int(np.diff(frames).max()) if len(grp) > 1 else 0
        cls_counts = grp["class_name"].value_counts()
        main_cls = cls_counts.index[0] if len(cls_counts) else "unknown"
        track_info[tid] = {
            "start": int(frames.min()),
            "end": int(frames.max()),
            "det": len(grp),
            "cls": main_cls,
            "max_gap": max_gap,
            "max_jump": float(jumps.max()) if len(jumps) else 0.0,
            "avg_jump": float(jumps.mean()) if len(jumps) else 0.0,
            "first": grp.iloc[0],
            "last": grp.iloc[-1],
        }

    # Per-track metrics table
    lines = ["=" * 80, "FAST TRACKING REPORT", f"Input: {args.csv}", "=" * 80]
    lines.append(f"Unique tracks: {len(track_info)}")
    lines.append("")
    lines.append("PER-TRACK METRICS (top 20 by detections)")
    lines.append("-" * 80)
    lines.append(
        f"{'track':<8}{'class':<12}{'start':<8}{'end':<8}{'det':<6}{'max_gap':<9}{'max_jump':<10}{'avg_jump':<10}"
    )
    top = sorted(track_info.items(), key=lambda x: x[1]["det"], reverse=True)[:20]
    for tid, s in top:
        lines.append(
            f"{tid:<8}{s['cls']:<12}{s['start']:<8}{s['end']:<8}{s['det']:<6}{s['max_gap']:<9}{s['max_jump']:<10.2f}{s['avg_jump']:<10.2f}"
        )
    lines.append("")

    # ID switch candidates
    lines.append("CANDIDATE ID SWITCHES")
    lines.append("-" * 80)
    switches = []
    tids = list(track_info.keys())
    # Precompute start/end arrays
    np.array([track_info[t]["start"] for t in tids])
    np.array([track_info[t]["end"] for t in tids])
    [track_info[t]["cls"] for t in tids]

    for i in tqdm(range(len(tids)), desc="Checking ID switches", unit="track"):
        tid_a = tids[i]
        s_a = track_info[tid_a]
        for j in range(len(tids)):
            if i == j:
                continue
  
... (truncated)
```

### scripts/diagnose_tracking.py

```python
#!/usr/bin/env python3
"""
Diagnose tracking instability by analyzing detections CSV.

For each track, computes:
  - frame span and number of detections
  - max frame gap between consecutive detections
  - max spatial jump (in pixels) between consecutive detections
  - average jump

Flags suspicious tracks: max_gap > 10 frames OR max_jump > 50 pixels.

Usage:
    python scripts/diagnose_tracking.py --csv outputs/petevents_bev_300_split_detections.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def analyze_track(group):
    """Analyze a single track group."""
    group = group.sort_values("frame")
    frames = group["frame"].values
    x = group["cx"].values
    y = group["cy"].values

    if len(group) < 2:
        return {
            "num_detections": len(group),
            "start_frame": frames[0] if len(group) else None,
            "end_frame": frames[-1] if len(group) else None,
            "max_gap": 0,
            "max_jump": 0.0,
            "avg_jump": 0.0,
            "flag": False,
        }

    gaps = np.diff(frames)
    max_gap = int(gaps.max()) if len(gaps) > 0 else 0

    dx = np.diff(x)
    dy = np.diff(y)
    jumps = np.sqrt(dx**2 + dy**2)
    max_jump = float(jumps.max()) if len(jumps) > 0 else 0.0
    avg_jump = float(jumps.mean()) if len(jumps) > 0 else 0.0

    flag = bool(max_gap > 10 or max_jump > 50.0)

    return {
        "num_detections": len(group),
        "start_frame": int(frames[0]),
        "end_frame": int(frames[-1]),
        "max_gap": max_gap,
        "max_jump": max_jump,
        "avg_jump": avg_jump,
        "flag": flag,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", default="outputs/petevents_bev_300_split_detections.csv"
    )
    parser.add_argument("--report", default="outputs/tracking_diagnosis.csv")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "track_id" not in df.columns or "frame" not in df.columns:
        raise ValueError("CSV must contain 'track_id' and 'frame' columns")

    rows = []
    for track_id, group in df.groupby("track_id"):
        info = analyze_track(group)
        info["track_id"] = track_id
        rows.append(info)

    report_df = pd.DataFrame(rows).sort_values("track_id")
    report_df.to_csv(args.report, index=False)

    flagged = report_df[report_df["flag"]]
    print(f"Total tracks analyzed: {len(report_df)}")
    print(f"Suspicious tracks flagged: {len(flagged)}")
    print(f"Report saved to {args.report}")

    if not flagged.empty:
        print("\nSuspicious track list (top 20):")
        print(flagged.head(20).to_string(index=False))
    else:
        print("No suspicious tracks found.")


if __name__ == "__main__":
    main()

```

## 7. Debug Tracking Video Script

```python
#!/usr/bin/env python3
"""
Debug tracking by drawing all detection boxes with track IDs on the video.

Usage:
    python scripts/debug_tracking_video.py \
        --csv outputs/petevents_bev_300_stricter_split_detections.csv \
        --video data/sample_data/traffic_video.mp4 \
        --start 0 --end 150 \
        --output outputs/tracking_debug_0_150.mp4
"""

import argparse

import cv2
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--video", default="data/sample_data/traffic_video.mp4")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=150)
    parser.add_argument("--output", default="outputs/tracking_debug.mp4")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.2,
        help="Only show boxes with confidence above this",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.csv)
    df = df[(df["frame"] >= args.start) & (df["frame"] <= args.end)]
    if df.empty:
        print("No detections in range")
        return

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print(f"Creating debug video: frames {args.start}-{args.end}, detections={len(df)}")

    for frame_idx in range(args.start, args.end + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        frame_dets = df[df["frame"] == frame_idx]
        for _, det in frame_dets.iterrows():
            if det.get("conf", 1.0) < args.conf:
                continu
... (truncated)
```

## 8. Tracking Output Sample

No tracking report CSV found in outputs/ (may have been cleaned).

## 9. Validation & CI Integration

No direct tracking-specific CI step (tracking is included in output validation).
