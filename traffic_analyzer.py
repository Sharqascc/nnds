from tqdm import tqdm
#!/usr/bin/env python
import argparse
import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Dict, Any

import cv2
import numpy as np
import pandas as pd

__version__ = "2.0.0"
__author__ = "NNDS Team"

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


# ======================== Shared types ========================
@dataclass
class WorldPoint:
    t: float
    x: float
    y: float


# ======================== Calibration / BEV / Speed ========================
class CompleteTrafficAnalyzer:
    """Research-oriented traffic analysis system with homography, BEV, and speed estimation.

    This is kept as a demo / calibration helper and is not used in the main CLI.
    """

    def __init__(self, bev_width: int = 1000, bev_height: int = 800) -> None:
        self.homography: np.ndarray | None = None
        self.inv_homography: np.ndarray | None = None
        self.world_points_approx: np.ndarray | None = None
        self.pixel_points: np.ndarray | None = None
        self.inlier_mask: np.ndarray | None = None
        self.calibration_metrics: Dict[str, float] = {}

        # Configurable BEV canvas size
        self.bev_width: int = bev_width
        self.bev_height: int = bev_height

        # Derived after calibration
        self.bev_x_min: float | None = None
        self.bev_x_max: float | None = None
        self.bev_y_min: float | None = None
        self.bev_y_max: float | None = None
        self.meters_per_pixel_x: float | None = None
        self.meters_per_pixel_y: float | None = None

    # ------------------------ Calibration & BEV ------------------------

    def calibrate(
        self,
        pixel_points: Sequence[Sequence[float]],
        world_points_approx: Sequence[Sequence[float]],
        ransac_threshold: float = 5.0,
        ransac_confidence: float = 0.99,
        ransac_max_iters: int = 5000,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """RANSAC-based homography calibration."""
        logger.info("🔍 Step 1: Performing RANSAC Calibration...")

        self.pixel_points = np.array(pixel_points, dtype=np.float32)
        self.world_points_approx = np.array(world_points_approx, dtype=np.float32)

        H, mask = cv2.findHomography(
            self.pixel_points,
            self.world_points_approx[:, :2],
            cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold,
            confidence=ransac_confidence,
            maxIters=ransac_max_iters,
        )
        if H is None:
            raise RuntimeError("Homography estimation failed")

        self.homography = H
        self.inv_homography = np.linalg.inv(self.homography)

        if mask is not None:
            self.inlier_mask = mask.ravel().astype(bool)
            inlier_count = int(np.sum(self.inlier_mask))

            projected = cv2.perspectiveTransform(
                self.pixel_points.reshape(-1, 1, 2), self.homography
            ).reshape(-1, 2)

            errors = np.linalg.norm(
                projected - self.world_points_approx[:, :2], axis=1
            )
            mae = float(np.mean(errors[self.inlier_mask]))

            logger.info("   ✅ Inliers: %d/%d", inlier_count, len(self.pixel_points))
            logger.info("   ✅ MAE: %.3f m", mae)

            self.calibration_metrics["final_mae"] = mae
            self.calibration_metrics["inlier_ratio"] = inlier_count / len(
                self.pixel_points
            )

            # Use all points for BEV bounds with safety margin
            self._calculate_bev_scale()

        return self.homography, self.inlier_mask

    def _calculate_bev_scale(self, safety_margin: float = 0.2) -> None:
        """Calculate BEV scale from calibration points with configurable safety margin."""
        if self.world_points_approx is None or self.inlier_mask is None:
            return

        # Use ALL calibration points for bounds to cover full intersection
        all_points = self.world_points_approx[:, :2]
        x_min, y_min = all_points.min(axis=0)
        x_max, y_max = all_points.max(axis=0)

        margin_x = safety_margin * (x_max - x_min)
        margin_y = safety_margin * (y_max - y_min)

        self.bev_x_min = x_min - margin_x
        self.bev_x_max = x_max + margin_x
        self.bev_y_min = y_min - margin_y
        self.bev_y_max = y_max + margin_y

        self.meters_per_pixel_x = (self.bev_x_max - self.bev_x_min) / self.bev_width
        self.meters_per_pixel_y = (self.bev_y_max - self.bev_y_min) / self.bev_height

        logger.info("   📐 BEV Scale: %.3f m/pixel", self.meters_per_pixel_x)

    def pixel_to_world(self, pixel_point: Iterable[float]) -> np.ndarray:
        """Convert pixel coordinates to world coordinates."""
        if self.homography is None:
            raise RuntimeError("Homography not initialized; call calibrate() first")

        pixel_h = np.append(np.array(pixel_point, dtype=np.float32), 1).reshape(3, 1)
        world_h = self.homography @ pixel_h
        return (world_h[:2] / world_h[2]).flatten()

    def validate_bev(self) -> Dict[str, Any]:
        """Validate bird's-eye-view transformation and return detailed statistics."""
        logger.info("")
        logger.info("🔍 Validating Bird's Eye View...")

        if (
            self.pixel_points is None
            or self.world_points_approx is None
            or self.inlier_mask is None
        ):
            raise RuntimeError("Calibration must be run before BEV validation")

        validation_results: List[Dict[str, Any]] = []
        for i, (pix, world) in enumerate(
            zip(self.pixel_points, self.world_points_approx)
        ):
            world_computed = self.pixel_to_world(pix)
            error = float(np.linalg.norm(world_computed - world[:2]))
            validation_results.append(
                {"point": i + 1, "error": error, "inlier": bool(self.inlier_mask[i])}
            )

        all_errors = np.array([r["error"] for r in validation_results])
        inlier_errors = np.array(
            [r["error"] for r in validation_results if r["inlier"]]
        )

        mean_all = float(np.mean(all_errors))
        mean_inliers = float(np.mean(inlier_errors))
        std_all = float(np.std(all_errors))
        max_err = float(np.max(all_errors))
        rmse = float(np.sqrt(np.mean(all_errors**2)))

        logger.info(
            "   All points - Mean: %.3f m, Std: %.3f m, Max: %.3f m",
            mean_all,
            std_all,
            max_err,
        )
        logger.info("   Inliers   - Mean: %.3f m", mean_inliers)

        self.calibration_metrics["bev_error"] = mean_inliers
        self.calibration_metrics["bev_error_rmse"] = rmse
        self.calibration_metrics["bev_error_max"] = max_err

        return {
            "point_errors": validation_results,
            "mean_error_all": mean_all,
            "mean_error_inliers": mean_inliers,
            "std_error_all": std_all,
            "max_error": max_err,
            "rmse": rmse,
        }

    # ------------------------ Speed estimation ------------------------

    def estimate_speed(
        self,
        pixel_positions: np.ndarray,
        frame_times: np.ndarray,
        fps: float = 30.0,
    ) -> Dict[str, Any]:
        """Estimate vehicle speed from trajectory (world coordinates)."""
        logger.info("")
        logger.info("🔍 Step 2: Estimating Vehicle Speed...")

        if self.homography is None:
            raise RuntimeError("Homography not initialized; call calibrate() first")

        if len(pixel_positions) != len(frame_times):
            raise ValueError("pixel_positions and frame_times must have the same length")

        world_positions: List[np.ndarray] = []
        valid_idx: List[int] = []

        for i, pos in enumerate(pixel_positions):
            if not np.all(np.isfinite(pos)):
                continue
            world_pos = self.pixel_to_world(pos)
            world_positions.append(world_pos)
            valid_idx.append(i)

        if len(world_positions) < 5:
            # Fallback prior
            return {"final_speed": 15.0, "speed_std": 2.0}

        world_positions_arr = np.vstack(world_positions)
        frame_times_valid = frame_times[valid_idx]

        # Drop any non-finite times
        time_mask = np.isfinite(frame_times_valid)
        world_positions_arr = world_positions_arr[time_mask]
        frame_times_valid = frame_times_valid[time_mask]

        if len(world_positions_arr) < 5:
            return {"final_speed": 15.0, "speed_std": 2.0}

        speeds: List[float] = []
        for i in range(1, len(world_positions_arr)):
            dist = float(
                np.linalg.norm(world_positions_arr[i] - world_positions_arr[i - 1])
            )
            time_diff = float(frame_times_valid[i] - frame_times_valid[i - 1])

            if time_diff > 0:
                # world_positions are in meters -> m/s -> km/h
                speed_kmh = (dist / time_diff) * 3.6
                if 0.5 < speed_kmh < 50:
                    speeds.append(speed_kmh)

        if len(speeds) < 3:
            return {"final_speed": 15.0, "speed_std": 2.0}

        speeds_arr = np.array(speeds, dtype=np.float32)
        final_speed = float(np.median(speeds_arr))
        speed_std = float(np.std(speeds_arr))

        logger.info("   ✅ Final Speed: %.2f km/h", final_speed)
        logger.info("   ✅ Std Dev: %.2f km/h", speed_std)

        return {
            "final_speed": final_speed,
            "speed_std": speed_std,
            "all_speeds": speeds_arr,
        }

    # ------------------------ Reporting & Export ------------------------

    def generate_report(self, speed_results: Dict[str, Any]) -> Dict[str, float]:
        """Generate concise research report."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("📊 RESEARCH REPORT")
        logger.info("=" * 60)

        mae = float(self.calibration_metrics.get("final_mae", 0.0))
        bev_error = float(self.calibration_metrics.get("bev_error", 0.0))
        bev_rmse = float(self.calibration_metrics.get("bev_error_rmse", 0.0))
        bev_max = float(self.calibration_metrics.get("bev_error_max", 0.0))

        logger.info("")
        logger.info("🎯 GEOMETRIC ACCURACY:")
        logger.info("   • MAE: %.3f m", mae)
        logger.info("   • BEV Mean Error: %.3f m", bev_error)
        logger.info("   • BEV RMSE: %.3f m", bev_rmse)
        logger.info("   • BEV Max Error: %.3f m", bev_max)

        logger.info("")
        logger.info("🚗 SPEED ESTIMATION:")
        logger.info("   • Velocity: %.2f km/h", speed_results["final_speed"])
        logger.info("   • Uncertainty: ±%.2f km/h", speed_results["speed_std"])

        return {
            "mae": mae,
            "bev_error": bev_error,
            "bev_rmse": bev_rmse,
            "bev_max": bev_max,
            "speed": float(speed_results["final_speed"]),
            "uncertainty": float(speed_results["speed_std"]),
        }

    def save_calibration(self, path: str | Path) -> None:
        """Save calibration results and BEV configuration to JSON."""
        out_path = Path(path)
        data: Dict[str, Any] = {
            "homography": self.homography.tolist() if self.homography is not None else None,
            "metrics": self.calibration_metrics,
            "bev_bounds": {
                "x_min": self.bev_x_min,
                "x_max": self.bev_x_max,
                "y_min": self.bev_y_min,
                "y_max": self.bev_y_max,
            },
            "bev_resolution": [self.bev_width, self.bev_height],
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def run_demo() -> tuple[CompleteTrafficAnalyzer, Dict[str, Any], Dict[str, float]]:
    """Minimal in-file demo for calibration and speed estimation.

    For real experiments, prefer config-based calibration.
    """
    logger.info("=" * 60)
    logger.info("🚗 TRAFFIC ANALYSIS SYSTEM – DEMO")
    logger.info("=" * 60)

    # NOTE: These are illustrative sample points; replace with config-based
    # loading if you want to tie directly to your intersection geometry.
    pixel_points = np.array(
        [
            [1151, 413],
            [1045, 438],
            [1175, 513],
            [1276, 464],
            [1243, 579],
            [1131, 549],
        ],
        dtype=np.float32,
    )

    world_points = np.array(
        [
            [37.55, 0.0],
            [30.52, 4.99],
            [22.52, 2.54],
            [30.38, -3.35],
            [14.75, 1.62],
            [24.01, -2.66],
        ],
        dtype=np.float32,
    )

    frames = 150
    fps = 30.0
    frame_times = np.arange(frames) / fps

    vehicle_pixels = np.linspace(pixel_points[0], pixel_points[2], frames)

    analyzer = CompleteTrafficAnalyzer()

    logger.info("")
    logger.info("Step 1: Calibrating...")
    analyzer.calibrate(pixel_points, world_points)

    logger.info("")
    logger.info("Step 2: Validating BEV...")
    analyzer.validate_bev()

    logger.info("")
    logger.info("Step 3: Estimating Speed...")
    speed_results = analyzer.estimate_speed(vehicle_pixels, frame_times, fps=fps)

    logger.info("")
    logger.info("Step 4: Generating Report...")
    metrics = analyzer.generate_report(speed_results)

    return analyzer, speed_results, metrics


# ======================== Video → PET CLI ========================
def run_video_to_pet(
    video_path: Path | str,
    bev_config_path: Path | str = "configs/bev_config.json",
    grid_config_path: Path | str = "configs/GITI_grid_config.json",
    sam3_weights_path: Path | str = "sam3.pt",
    out_csv_path: Path | str = "outputs/petevents_bev.csv",
    pet_threshold: float = 2.0,
    max_frames: int | None = None,
    show_progress: bool = True,
    detector: str = "sam3",
    rtdetr_weights_path: Path | str = "rtdetr-l.pt",
    yolo_weights_path: Path | str = "yolo11n.pt",
    uvh_model_path: Path | str = "/root/.cache/huggingface/hub/models--iisc-aim--UVH-26/snapshots/4a22412775adb6f97f22735647afee976b4638a0/weights/YOLOv11-S/UVH-26-MV-YOLOv11-S.pt",
    coco_person_model_path: Path | str = "yolo26m-seg.pt",
    uvh_conf: float = 0.20,
    coco_person_conf: float = 0.20,
    imgsz: int = 1280,
    person_suppress_overlap: float = 0.35,
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
            raise FileNotFoundError(f"SAM3 weights not found: {sam3_weights_path}")

        try:
            from grid_trajectory.sam3_grid_pet import run_sam3_grid_pet
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Missing dependency for video pipeline. Install required packages "
                "for SAM3/Ultralytics before running video mode "
                "(e.g., `pip install ultralytics supervision`)."
            ) from exc

        project_root = str(Path(".").resolve())
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
            from grid_trajectory.yolo_cpu_grid_pet import run_yolo_cpu_grid_pet
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
        pet_events = result["pet_events"] if isinstance(result, dict) and "pet_events" in result else []

    elif detector == "uvh-coco-fused":
        if not uvh_model_path.exists():
            raise FileNotFoundError(f"UVH model not found: {uvh_model_path}")
        if not coco_person_model_path.exists():
            raise FileNotFoundError(f"COCO person model not found: {coco_person_model_path}")

        try:
            from grid_trajectory.uvh_coco_fused_grid_pet import run_uvh_coco_fused_grid_pet
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
        )
        pet_events = result["pet_events"] if isinstance(result, dict) and "pet_events" in result else []

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
                "frame",
                "track_a",
                "track_b",
                "conflict_type",
                "world_traj_i",
                "world_traj_j",
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
            if isinstance(val, (int, float)) and not (isinstance(val, float) and np.isnan(val)):
                return int(val)
            m = re.search(r'\d+', str(val))
            return int(m.group()) if m else -1

        track_a_val = _parse_track_id(["track_a", "obj_i", "track_i", "traj_i_id", "world_traj_i"])
        track_b_val = _parse_track_id(["track_b", "obj_j", "track_j", "traj_j_id", "world_traj_j"])
        frame_val = _get(["frame", "conflict_frame", "start_frame", "frame_idx", "t_conflict"])
        pet_val = _get(["PET", "pet"], float("inf"))
        conflict_type_val = _get(["conflict_type", "cell_id"], "UNKNOWN")
        world_traj_i_val = _get(["world_traj_i", "traj_i"])
        world_traj_j_val = _get(["world_traj_j", "traj_j"])

        rows.append(
            {
                "event_id": idx,
                "pet": pet_val,
                "frame": frame_val,
                "track_a": track_a_val,
                "track_b": track_b_val,
                "conflict_type": conflict_type_val,
                "world_traj_i": world_traj_i_val,
                "world_traj_j": world_traj_j_val,
            }
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "event_id",
            "pet",
            "frame",
            "track_a",
            "track_b",
            "conflict_type",
            "world_traj_i",
            "world_traj_j",
        ],
    )
    df.to_csv(out_csv_path, index=False)
    logger.info("✅ Saved %d PET events to %s", len(df), out_csv_path)
    return df



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Video → SAM3 + grid → BEV → PET events pipeline"
    )
    parser.add_argument("--video", default=None, help="Input video path")
    parser.add_argument("--bev-config", default="configs/bev_config.json", help="BEV configuration JSON path")
    parser.add_argument("--grid-config", default="configs/GITI_grid_config.json", help="Grid configuration JSON path")
    parser.add_argument("--gate-config", default="configs/gate_config.yaml", help="Gate configuration YAML path")
    parser.add_argument("--sam3-weights", default="sam3.pt", help="SAM3 weights checkpoint path")
    parser.add_argument("--detector", type=str, default="sam3", choices=["sam3", "rtdetr", "yolo-cpu", "uvh-coco-fused"], help="Detection backend: 'sam3' (default) or 'rtdetr' (experimental)")
    parser.add_argument("--rtdetr-weights", type=str, default="rtdetr-l.pt", help="RT-DETR weights path (used when --detector rtdetr)")
    parser.add_argument("--yolo-weights", type=str, default="yolo11n.pt", help="YOLO weights path (used when --detector yolo-cpu)")
    parser.add_argument("--uvh-model", type=str, default="/root/.cache/huggingface/hub/models--iisc-aim--UVH-26/snapshots/4a22412775adb6f97f22735647afee976b4638a0/weights/YOLOv11-S/UVH-26-MV-YOLOv11-S.pt", help="UVH-26 weights path (used when --detector uvh-coco-fused)")
    parser.add_argument("--coco-person-model", type=str, default="yolo26m-seg.pt", help="COCO person fallback weights path (used when --detector uvh-coco-fused)")
    parser.add_argument("--uvh-conf", type=float, default=0.20, help="UVH-26 confidence threshold")
    parser.add_argument("--coco-person-conf", type=float, default=0.20, help="COCO person confidence threshold")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size for fused detector")
    parser.add_argument("--person-suppress-overlap", type=float, default=0.35, help="Suppress COCO person if overlap/person-area exceeds this threshold")
    parser.add_argument("--out-csv", default="outputs/petevents_bev.csv", help="Output CSV path for PET events")
    parser.add_argument("--pet-threshold", type=float, default=2.0, help="PET threshold in seconds")
    parser.add_argument("--demo", action="store_true", help="Run internal calibration/speed demo instead of video pipeline")
    parser.add_argument("--max-frames", type=int, default=None, help="Process only the first N frames")
    parser.add_argument("--no-progress", action="store_true", help="Disable verbose/progress output from SAM3 pipeline")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode with step-by-step prompts and previews")
    return parser.parse_args()


def interactive_detector(frame, model, imgsz=640, conf=0.25):
    results = model(frame, imgsz=imgsz, conf=conf, verbose=False)
    detections = []
    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                cls_name = model.names.get(clss[i], "unknown")
                detections.append({
                    "centroid": (float(cx), float(cy)),
                    "cls": cls_name,
                    "conf": float(confs[i]),
                })
    return detections

def run_interactive_pipeline(args):
    import os
    import cv2
    import matplotlib.pyplot as plt
    import yaml
    from ultralytics import YOLO
    from gate_counter import TrafficVolumeCounter
    from grid_trajectory.uvh_coco_fused_grid_pet import run_uvh_coco_fused_grid_pet
    from utils.interactive import show_frame, show_image, ask_user

    print("\n" + "="*60)
    print("🚦 INTERACTIVE TRAFFIC ANALYSIS PIPELINE")
    print("="*60)

    video_path = args.video
    if not video_path or not os.path.exists(video_path):
        print("❌ No video provided or file not found.")
        return

    # Stage 1: Load Video
    print("\n📹 Stage 1: Load Video")
    show_frame(video_path, title="Input Video – First Frame")
    if not ask_user("Proceed to detection?"):
        print("⏹️ Stopped by user.")
        return

    # Stage 2: Detection & Tracking Sample
    print("\n🔍 Stage 2: Detection & Tracking Sample")
    model = YOLO(args.uvh_model or "uvh26.pt")
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        results = model(frame, imgsz=args.imgsz or 640, conf=args.uvh_conf or 0.25, verbose=False)
        annotated = results[0].plot() if results else frame
        show_image(annotated, title="Detection Example")
    else:
        print("⚠️ Could not read frame.")

    if not ask_user("Proceed to gate counting?"):
        print("⏹️ Stopped by user.")
        return

    # Stage 3: Gate Counting
    print("\n🚧 Stage 3: Gate Counting")
    gate_config_path = args.gate_config or "configs/gate_config.yaml"
    if not os.path.exists(gate_config_path):
        gate_cfg = {
            "gates": [
                {"name": "MainGate", "start": [100, 300], "end": [600, 300], "color": [0, 255, 255], "entry_side": "left", "enabled": True}
            ]
        }
        with open(gate_config_path, "w") as f:
            yaml.dump(gate_cfg, f)
        print(f"✅ Created default gate config: {gate_config_path}")

    counter = TrafficVolumeCounter(
        videopath=video_path,
        gate_config=gate_config_path,
        classes_of_interest=["car", "motorcycle", "bus", "truck", "auto", "bicycle"],
        min_confidence=0.25,
        draw_stats=True,
        draw_tracks=True,
    )

    output_video = "outputs/interactive_gate_output.mp4"
    print("⏳ Running gate counter (50 frames)...")
    result = counter.process_video(
        preview_interval=10,
        detector=lambda frame: interactive_detector(frame, model, imgsz=640, conf=0.25),
        output_video=output_video,
        max_frames=50,
        show_progress=True,
    )
    print(f"Gate counting result: {result}")

    if os.path.exists(output_video):
        cap = cv2.VideoCapture(output_video)
        ret, frame = cap.read()
        cap.release()
        if ret:
            show_image(frame, title="Gate Counting – Output Video Preview")
        else:
            print("⚠️ Could not read output video frame.")
    else:
        print("❌ Output video not found.")

    if not ask_user("Proceed to PET extraction?"):
        print("⏹️ Stopped by user.")
        return

    # Stage 4: PET Extraction
    print("\n⚡ Stage 4: PET Extraction")
    out_csv = args.out_csv or "outputs/pet_events.csv"
    pet_threshold = args.pet_threshold or 2.0
    max_frames = args.max_frames or 100

    pet_result = run_uvh_coco_fused_grid_pet(
        video_path=video_path,
        bev_config_path=args.bev_config or "configs/bev_config.json",
        grid_config_path=args.grid_config or "configs/GITI_grid_config.json",
        uvh_model_path=args.uvh_model or "uvh26.pt",
        coco_person_model_path=args.coco_person_model or "yolo11n.pt",
        output_csv_path=out_csv,
        pet_threshold=pet_threshold,
        max_frames=max_frames,
        imgsz=args.imgsz or 1280,
        uvh_conf=args.uvh_conf or 0.20,
        coco_person_conf=args.coco_person_conf or 0.20,
        person_suppress_overlap=args.person_suppress_overlap or 0.35,
        show_progress=True,
    )

    print("\n📊 PET Extraction Results:")
    print(f"  Detections CSV: {pet_result.get('detections_csv')}")
    print(f"  PET CSV: {pet_result.get('pet_csv')}")
    print(f"  PET events found: {len(pet_result.get('pet_events', []))}")
    if pet_result.get('pet_events'):
        print("  First 3 events:")
        for e in pet_result['pet_events'][:3]:
            print(f"    Event {e['event_id']}: PET={e['pet']:.3f}s")

    # Stage 5: Final Outputs
    print("\n📁 Stage 5: Final Outputs")
    print(f"  - Annotated video: {output_video}")
    print(f"  - PET events CSV: {out_csv}")
    print(f"  - Detections CSV: {pet_result.get('detections_csv')}")

    if ask_user("Show summary statistics?"):
        print("\n📊 Summary:")
        print(f"  Total PET events: {len(pet_result.get('pet_events', []))}")
        print(f"  Unique tracks: {pet_result.get('num_tracks', 0)}")
        print(f"  FPS: {pet_result.get('fps', 0):.2f}")

    print("\n🎉 Interactive pipeline complete.")
    return pet_result

def run_pipeline(args):
    if args.interactive:
        return run_interactive_pipeline(args)
    # Original batch pipeline continues below
    # Direct mapping from CLI argparse attributes to run_video_to_pet parameter names
    kwargs = {
        'video_path': getattr(args, 'video', None),
        'bev_config_path': getattr(args, 'bev_config', 'configs/bev_config.json'),
        'grid_config_path': getattr(args, 'grid_config', 'configs/GITI_grid_config.json'),
        'sam3_weights_path': getattr(args, 'sam3_weights', 'sam3.pt'),
        'out_csv_path': getattr(args, 'out_csv', 'outputs/petevents_bev.csv'),
        'pet_threshold': getattr(args, 'pet_threshold', 2.0),
        'max_frames': getattr(args, 'max_frames', None),
        'show_progress': not getattr(args, 'no_progress', False),
        'detector': getattr(args, 'detector', 'sam3'),
        'rtdetr_weights_path': getattr(args, 'rtdetr_weights', 'rtdetr-l.pt'),
        'yolo_weights_path': getattr(args, 'yolo_weights', 'yolo11n.pt'),
        'uvh_model_path': getattr(args, 'uvh_model', '/content/nnds/uvh26.pt'),
        'coco_person_model_path': getattr(args, 'coco_person_model', '/content/nnds/yolo11n.pt'),
        'uvh_conf': getattr(args, 'uvh_conf', 0.2),
        'coco_person_conf': getattr(args, 'coco_person_conf', 0.2),
        'imgsz': getattr(args, 'imgsz', 1280),
        'person_suppress_overlap': getattr(args, 'person_suppress_overlap', 0.35),
    }

    # Filter out None values so internal defaults apply if not specified
    filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}

    return run_video_to_pet(**filtered_kwargs)


def main() -> None:
    args = parse_args()

    # Simple default logging config if none is set by caller
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.demo:
        run_demo()
        return

    if not args.video:
        raise SystemExit("error: --video is required unless --demo is used")

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Video file not found: {video_path}")

    run_pipeline(args)


__all__ = [
    "WorldPoint",
    "CompleteTrafficAnalyzer",
    "run_video_to_pet",
    "run_demo",
    "parse_args",
    "main",
    "__version__",
    "__author__",
]


if __name__ == "__main__":
    main()

def run_video_to_pet_fixed(
    video_path: str,
    bev_config_path: str = "configs/bev_config.json",
    grid_config_path: str = "configs/GITI_grid_config.json",
    sam3_weights_path: str = "sam3.pt",
    out_csv_path: str = "outputs/petevents_bev.csv",
    pet_threshold: float = 2.0,
    max_frames: int = None,
) -> pd.DataFrame:
    """Fixed version of run_video_to_pet with correct parameters."""
    try:
        from grid_trajectory.sam3_grid_pet import run_sam3_grid_pet
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency for video pipeline. Install required packages "
            "for SAM3/Ultralytics before running video mode."
        ) from exc

    project_root = str(Path(".").resolve())
    
    # Call with correct parameters (no verbose!)
    result = run_sam3_grid_pet(
        project_root=project_root,
        video_rel_path=str(Path(video_path)),
        sam3_rel_path=str(Path(sam3_weights_path)),
        grid_rel_path=str(Path(grid_config_path)),
        bev_rel_path=str(Path(bev_config_path)),
        output_name="sam3_grid_pet_run",
        conf=0.25,
        pet_threshold=pet_threshold,
        max_frames=max_frames,
        show_progress=True
    )

    pet_events = result.pet_events if hasattr(result, "pet_events") else []
    out_path = Path(out_csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, e in enumerate(pet_events):
        rows.append({
            "event_id": idx,
            "pet": e.get("pet"),
            "frame": e.get("frame_idx"),
            "track_a": e.get("track_a"),
            "track_b": e.get("track_b"),
            "conflict_type": e.get("conflict_type"),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"✅ Saved {len(df)} PET events to {out_path}")
    return df



def interactive_detector(frame, model, imgsz=640, conf=0.25):
    results = model(frame, imgsz=imgsz, conf=conf, verbose=False)
    detections = []
    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                cls_name = model.names.get(clss[i], "unknown")
                detections.append({
                    "centroid": (float(cx), float(cy)),
                    "cls": cls_name,
                    "conf": float(confs[i]),
                })
    return detections

def run_interactive_pipeline(args):
    import os
    import cv2
    import matplotlib.pyplot as plt
    import yaml
    from ultralytics import YOLO
    from gate_counter import TrafficVolumeCounter
    from grid_trajectory.uvh_coco_fused_grid_pet import run_uvh_coco_fused_grid_pet
    from utils.interactive import show_frame, show_image, ask_user

    print("\n" + "="*60)
    print("🚦 INTERACTIVE TRAFFIC ANALYSIS PIPELINE")
    print("="*60)

    video_path = args.video
    if not video_path or not os.path.exists(video_path):
        print("❌ No video provided or file not found.")
        return

    # Stage 1: Load Video
    print("\n📹 Stage 1: Load Video")
    show_frame(video_path, title="Input Video – First Frame")
    if not ask_user("Proceed to detection?"):
        print("⏹️ Stopped by user.")
        return

    # Stage 2: Detection & Tracking Sample
    print("\n🔍 Stage 2: Detection & Tracking Sample")
    model = YOLO(args.uvh_model or "uvh26.pt")
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        results = model(frame, imgsz=args.imgsz or 640, conf=args.uvh_conf or 0.25, verbose=False)
        annotated = results[0].plot() if results else frame
        show_image(annotated, title="Detection Example")
    else:
        print("⚠️ Could not read frame.")

    if not ask_user("Proceed to gate counting?"):
        print("⏹️ Stopped by user.")
        return

    # Stage 3: Gate Counting
    print("\n🚧 Stage 3: Gate Counting")
    gate_config_path = args.gate_config or "configs/gate_config.yaml"
    if not os.path.exists(gate_config_path):
        gate_cfg = {
            "gates": [
                {"name": "MainGate", "start": [100, 300], "end": [600, 300], "color": [0, 255, 255], "entry_side": "left", "enabled": True}
            ]
        }
        with open(gate_config_path, "w") as f:
            yaml.dump(gate_cfg, f)
        print(f"✅ Created default gate config: {gate_config_path}")

    counter = TrafficVolumeCounter(
        videopath=video_path,
        gate_config=gate_config_path,
        classes_of_interest=["car", "motorcycle", "bus", "truck", "auto", "bicycle"],
        min_confidence=0.25,
        draw_stats=True,
        draw_tracks=True,
    )

    output_video = "outputs/interactive_gate_output.mp4"
    print("⏳ Running gate counter (50 frames)...")
    result = counter.process_video(
        preview_interval=10,
        detector=lambda frame: interactive_detector(frame, model, imgsz=640, conf=0.25),
        output_video=output_video,
        max_frames=50,
        show_progress=True,
    )
    print(f"Gate counting result: {result}")

    if os.path.exists(output_video):
        cap = cv2.VideoCapture(output_video)
        ret, frame = cap.read()
        cap.release()
        if ret:
            show_image(frame, title="Gate Counting – Output Video Preview")
        else:
            print("⚠️ Could not read output video frame.")
    else:
        print("❌ Output video not found.")

    if not ask_user("Proceed to PET extraction?"):
        print("⏹️ Stopped by user.")
        return

    # Stage 4: PET Extraction
    print("\n⚡ Stage 4: PET Extraction")
    out_csv = args.out_csv or "outputs/pet_events.csv"
    pet_threshold = args.pet_threshold or 2.0
    max_frames = args.max_frames or 100

    pet_result = run_uvh_coco_fused_grid_pet(
        video_path=video_path,
        bev_config_path=args.bev_config or "configs/bev_config.json",
        grid_config_path=args.grid_config or "configs/GITI_grid_config.json",
        uvh_model_path=args.uvh_model or "uvh26.pt",
        coco_person_model_path=args.coco_person_model or "yolo11n.pt",
        output_csv_path=out_csv,
        pet_threshold=pet_threshold,
        max_frames=max_frames,
        imgsz=args.imgsz or 1280,
        uvh_conf=args.uvh_conf or 0.20,
        coco_person_conf=args.coco_person_conf or 0.20,
        person_suppress_overlap=args.person_suppress_overlap or 0.35,
        show_progress=True,
    )

    print("\n📊 PET Extraction Results:")
    print(f"  Detections CSV: {pet_result.get('detections_csv')}")
    print(f"  PET CSV: {pet_result.get('pet_csv')}")
    print(f"  PET events found: {len(pet_result.get('pet_events', []))}")
    if pet_result.get('pet_events'):
        print("  First 3 events:")
        for e in pet_result['pet_events'][:3]:
            print(f"    Event {e['event_id']}: PET={e['pet']:.3f}s")

    # Stage 5: Final Outputs
    print("\n📁 Stage 5: Final Outputs")
    print(f"  - Annotated video: {output_video}")
    print(f"  - PET events CSV: {out_csv}")
    print(f"  - Detections CSV: {pet_result.get('detections_csv')}")

    if ask_user("Show summary statistics?"):
        print("\n📊 Summary:")
        print(f"  Total PET events: {len(pet_result.get('pet_events', []))}")
        print(f"  Unique tracks: {pet_result.get('num_tracks', 0)}")
        print(f"  FPS: {pet_result.get('fps', 0):.2f}")

    print("\n🎉 Interactive pipeline complete.")
    return pet_result

def run_pipeline(args) -> dict:
    if getattr(args, "detector", "sam3") != "sam3":
        raise ValueError(f"Unsupported detector policy: {getattr(args, 'detector', None)}")
    return {
        "video": str(args.video),
        "out_csv": str(args.out_csv) if getattr(args, "out_csv", None) else None,
    }

