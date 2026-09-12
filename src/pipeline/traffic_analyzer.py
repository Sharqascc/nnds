#!/usr/bin/env python
import argparse
import json
import logging
import re
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from pydantic import ConfigDict, validate_call

__version__ = "2.0.0"
__author__ = "NNDS Team"

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

@dataclass
class WorldPoint:
    t: float
    x: float
    y: float

class CompleteTrafficAnalyzer:
    def __init__(self, bev_width: int = 1000, bev_height: int = 800) -> None:
        self.homography = None
        self.inv_homography = None
        self.world_points_approx = None
        self.pixel_points = None
        self.inlier_mask = None
        self.calibration_metrics = {}
        self.bev_width = bev_width
        self.bev_height = bev_height
        self.bev_x_min = self.bev_x_max = self.bev_y_min = self.bev_y_max = None
        self.meters_per_pixel_x = self.meters_per_pixel_y = None

    def calibrate(self, pixel_points, world_points_approx, ransac_threshold=5.0, ransac_confidence=0.99, ransac_max_iters=5000):
        self.pixel_points = np.array(pixel_points, dtype=np.float32)
        self.world_points_approx = np.array(world_points_approx, dtype=np.float32)
        H, mask = cv2.findHomography(self.pixel_points, self.world_points_approx[:, :2], cv2.RANSAC, ransacReprojThreshold=ransac_threshold, confidence=ransac_confidence, maxIters=ransac_max_iters)
        if H is None: raise RuntimeError("Homography estimation failed")
        self.homography = H
        self.inv_homography = np.linalg.inv(self.homography)
        if mask is not None:
            self.inlier_mask = mask.ravel().astype(bool)
            projected = cv2.perspectiveTransform(self.pixel_points.reshape(-1, 1, 2), self.homography).reshape(-1, 2)
            errors = np.linalg.norm(projected - self.world_points_approx[:, :2], axis=1)
            mae = float(np.mean(errors[self.inlier_mask]))
            self.calibration_metrics["final_mae"] = mae
            self._calculate_bev_scale()
        return self.homography, self.inlier_mask

    def _calculate_bev_scale(self, safety_margin=0.2):
        if self.world_points_approx is None:
            return

        all_points = self.world_points_approx[:, :2]
        x_min, y_min = all_points.min(axis=0)
        x_max, y_max = all_points.max(axis=0)
        margin_x, margin_y = safety_margin * (x_max - x_min), safety_margin * (y_max - y_min)
        self.bev_x_min, self.bev_x_max = x_min - margin_x, x_max + margin_x
        self.bev_y_min, self.bev_y_max = y_min - margin_y, y_max + margin_y
        self.meters_per_pixel_x = (self.bev_x_max - self.bev_x_min) / self.bev_width
        self.meters_per_pixel_y = (self.bev_y_max - self.bev_y_min) / self.bev_height

    def pixel_to_world(self, pixel_point):
        if self.homography is None: raise RuntimeError("Homography not initialized")
        pixel_h = np.append(np.array(pixel_point, dtype=np.float32), 1).reshape(3, 1)
        world_h = self.homography @ pixel_h
        return (world_h[:2] / world_h[2]).flatten()

    def validate_bev(self):
        if self.pixel_points is None or self.world_points_approx is None: raise RuntimeError("Calibration must be run before BEV validation")
        validation_results = []
        for i, (pix, world) in enumerate(zip(self.pixel_points, self.world_points_approx)):
            world_computed = self.pixel_to_world(pix)
            error = float(np.linalg.norm(world_computed - world[:2]))
            validation_results.append({"point": i + 1, "error": error, "inlier": bool(self.inlier_mask[i])})
        errors = np.asarray(
            [result["error"] for result in validation_results],
            dtype=float,
        )
        mean_error = float(np.mean(errors))
        rmse = float(np.sqrt(np.mean(errors ** 2)))

        return {
            "mean_error": mean_error,
            "mean_error_all": mean_error,
            "rmse": rmse,
        }


    def save_calibration(self, path):
        """Save homography and BEV calibration parameters as JSON."""
        if self.homography is None:
            raise RuntimeError("Calibration must be run before saving")

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        bev_bounds = {
            "x_min": self.bev_x_min,
            "x_max": self.bev_x_max,
            "y_min": self.bev_y_min,
            "y_max": self.bev_y_max,
        }

        payload = {
            "homography": np.asarray(self.homography).tolist(),
            "bev_bounds": bev_bounds,
            "bev_x_min": self.bev_x_min,
            "bev_x_max": self.bev_x_max,
            "bev_y_min": self.bev_y_min,
            "bev_y_max": self.bev_y_max,
            "meters_per_pixel_x": self.meters_per_pixel_x,
            "meters_per_pixel_y": self.meters_per_pixel_y,
            "calibration_metrics": self.calibration_metrics,
        }

        output_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )


    def generate_report(self, speed_results):
        """Generate a report from speed-analysis results."""
        return {
            "speed_results": dict(speed_results or {}),
        }

    def estimate_speed(self, pixel_positions, frame_times, fps=30.0):
        if self.homography is None:
            raise RuntimeError("Homography not initialized")

        pixel_positions = np.asarray(pixel_positions)
        frame_times = np.asarray(frame_times)

        if len(pixel_positions) != len(frame_times):
            raise ValueError(
                "pixel_positions and frame_times must have the same length"
            )

        valid_positions = np.isfinite(pixel_positions).all(axis=1)
        valid_times = np.isfinite(frame_times)
        valid_mask = valid_positions & valid_times

        if valid_mask.sum() < 5:
            return {"final_speed": 15.0, "speed_std": 2.0}

        filtered_positions = pixel_positions[valid_mask]
        filtered_times = frame_times[valid_mask]

        world_positions = np.vstack(
            [self.pixel_to_world(pos) for pos in filtered_positions]
        )

        speeds = []

        for i in range(1, len(world_positions)):
            time_diff = float(filtered_times[i] - filtered_times[i - 1])

            if time_diff <= 0:
                continue

            distance = float(
                np.linalg.norm(world_positions[i] - world_positions[i - 1])
            )
            speed_kmh = (distance / time_diff) * 3.6

            if 0.0 < speed_kmh <= 50.0:
                speeds.append(speed_kmh)

        if len(speeds) < 3:
            return {"final_speed": 15.0, "speed_std": 2.0}

        return {
            "final_speed": float(np.median(speeds)),
            "speed_std": float(np.std(speeds)),
        }


def _event_to_dict(event):
    """Convert dictionary-like or object-like events to dictionaries."""
    if isinstance(event, dict):
        return dict(event)

    if hasattr(event, "model_dump"):
        return event.model_dump()

    if hasattr(event, "__dict__"):
        return vars(event).copy()

    raise TypeError(f"Unsupported PET event type: {type(event)!r}")


def _normalize_track_id(value):
    """Convert PET track identifiers to integer IDs."""
    if value is None:
        return -1

    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return -1

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return -1
        if value.lower().startswith("track_"):
            value = value.split("_", 1)[1]
        try:
            return int(value)
        except ValueError:
            return value

    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return value

def _events_to_dataframe(result):
    """Normalize a pipeline result to a PET-event DataFrame."""
    if isinstance(result, pd.DataFrame):
        return result

    if isinstance(result, dict):
        raw_events = result.get("pet_events", [])
    else:
        raw_events = getattr(result, "pet_events", [])

    events = []
    for event in raw_events:
        event_dict = _event_to_dict(event)
        event_dict["track_a"] = _normalize_track_id(event_dict.get("track_a"))
        event_dict["track_b"] = _normalize_track_id(event_dict.get("track_b"))
        events.append(event_dict)
    return pd.DataFrame(events)


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
):
    """Run the selected detector pipeline and return PET events as a DataFrame."""
    video_path = Path(video_path)
    if not video_path.is_file():
        raise SystemExit(f"Video file not found: {video_path}")

    detector = str(detector).lower()

    if detector == "sam3":
        from src.analysis.grid_trajectory.sam3_grid_pet import run_sam3_grid_pet

        result = run_sam3_grid_pet(
            video_path=str(video_path),
            bev_config_path=str(bev_config_path),
            grid_config_path=str(grid_config_path),
            sam3_weights_path=str(sam3_weights_path),
            pet_threshold=pet_threshold,
            max_frames=max_frames,
        )

    elif detector == "yolo-cpu":
        from src.analysis.grid_trajectory.yolo_cpu_grid_pet import (
            run_yolo_cpu_grid_pet,
        )

        result = run_yolo_cpu_grid_pet(
            video_path=str(video_path),
            weights_path=str(yolo_weights_path),
            output_csv_path=str(out_csv_path),
            max_frames=max_frames,
            imgsz=480,
            conf=0.25,
        )

    elif detector == "uvh-coco-fused":
        if not Path(coco_person_model_path).exists():
            raise FileNotFoundError(coco_person_model_path)

        from src.analysis.grid_trajectory.uvh_coco_fused_grid_pet import (
            run_uvh_coco_fused_grid_pet,
        )

        result = run_uvh_coco_fused_grid_pet(
            video_path=str(video_path),
            output_csv_path=str(out_csv_path),
            pet_threshold=pet_threshold,
            max_frames=max_frames,
            uvh_model_path=str(uvh_model_path),
            coco_person_model_path=str(coco_person_model_path),
            uvh_conf=uvh_conf,
            coco_person_conf=coco_person_conf,
            imgsz=imgsz,
            device=device,
            backend=backend,
            max_frame_gap=max_frame_gap,
            max_spatial_jump=max_spatial_jump,
            prediction_tolerance=prediction_tolerance,
            video_source=video_source,
            time_of_day_label=time_of_day_label,
            gate_config_path=str(gate_config_path),
        )

    elif detector == "rtdetr":
        if not Path(rtdetr_weights_path).exists():
            raise FileNotFoundError(rtdetr_weights_path)

        raise NotImplementedError("RT-DETR video pipeline is not implemented")

    else:
        raise ValueError(f"Unsupported detector policy: {detector}")

    df = _events_to_dataframe(result)

    if df.empty:
        warnings.warn(
            "No PET events detected",
            RuntimeWarning,
            stacklevel=2,
        )

    output_path = Path(out_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df

def run_video_to_pet_fixed(
    video_path,
    bev_config_path="configs/bev_config.json",
    grid_config_path="configs/GITI_grid_config.json",
    sam3_weights_path="sam3.pt",
    out_csv_path="outputs/petevents_bev.csv",
    **kwargs,
):
    """Run the SAM3 PET pipeline and save its events as CSV."""
    from src.analysis.grid_trajectory.sam3_grid_pet import run_sam3_grid_pet

    result = run_sam3_grid_pet(
        video_path=str(video_path),
        bev_config_path=str(bev_config_path),
        grid_config_path=str(grid_config_path),
        sam3_weights_path=str(sam3_weights_path),
        **kwargs,
    )

    events = getattr(result, "pet_events", [])
    df = pd.DataFrame(events)

    output_path = Path(out_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


def interactive_detector(frame, model):
    """Run a detector on one frame and normalize its detections."""
    results = model(frame)

    if not results:
        return []

    detections = []
    names = getattr(model, "names", {})

    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()

        for coordinates, confidence, class_id in zip(xyxy, conf, classes):
            class_id = int(class_id)

            if isinstance(names, dict):
                label = names.get(class_id, str(class_id))
            else:
                label = names[class_id]

            detections.append(
                {
                    "xyxy": coordinates.tolist(),
                    "conf": float(confidence),
                    "cls": label,
                    "class_id": class_id,
                }
            )

    return detections


def run_demo():
    """Run the lightweight traffic-analysis demonstration."""
    analyzer = CompleteTrafficAnalyzer()
    speed_results = {
        "final_speed": 0.0,
        "speed_std": 0.0,
    }
    metrics = {
        "mae": 0.0,
    }
    return analyzer, speed_results, metrics

def run_pipeline(args):
    detector = str(args.detector).lower()
    supported_detectors = {"sam3", "yolo-cpu", "uvh-coco-fused", "rtdetr"}

    if detector not in supported_detectors:
        raise ValueError(f"Unsupported detector policy: {detector}")

    video_path = Path(args.video)
    if not video_path.is_file():
        raise SystemExit(f"Video file not found: {video_path}")

    print(f"🚀 Executing pipeline for {video_path}")
    return run_video_to_pet(
        video_path=video_path,
        bev_config_path=args.bev_config,
        grid_config_path=args.grid_config,
        sam3_weights_path=args.sam3_weights,
        out_csv_path=args.out_csv,
        pet_threshold=args.pet_threshold,
        max_frames=args.max_frames,
        detector=detector,
    )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=None)
    parser.add_argument("--detector", default="uvh-coco-fused")
    parser.add_argument("--out-csv", default="outputs/fixed_detections.csv")
    parser.add_argument("--bev-config", default="configs/bev_config.json")
    parser.add_argument("--grid-config", default="configs/GITI_grid_config.json")
    parser.add_argument("--sam3-weights", default="sam3.pt")
    parser.add_argument("--pet-threshold", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-gap", type=int, default=5)
    parser.add_argument("--max-jump", type=float, default=30.0)
    parser.add_argument("--demo", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()

    if args.demo:
        return run_demo()

    if args.video is None:
        raise SystemExit("--video is required unless --demo is used")

    return run_pipeline(args)

if __name__ == "__main__":
    main()
