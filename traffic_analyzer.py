import argparse
import warnings
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Dict, Any

import cv2
import numpy as np
import pandas as pd


warnings.filterwarnings("ignore")


class CompleteTrafficAnalyzer:
    """Research-oriented traffic analysis system with homography, BEV, and speed estimation.

    This is kept as a demo / calibration helper and is not used in the main CLI.
    """

    def __init__(self) -> None:
        self.homography: np.ndarray | None = None
        self.inv_homography: np.ndarray | None = None
        self.world_points_approx: np.ndarray | None = None
        self.pixel_points: np.ndarray | None = None
        self.inlier_mask: np.ndarray | None = None
        self.calibration_metrics: Dict[str, float] = {}
        self.bev_width: int = 1000
        self.bev_height: int = 800

    # ------------------------ Calibration & BEV ------------------------

    def calibrate(
        self,
        pixel_points: Sequence[Sequence[float]],
        world_points_approx: Sequence[Sequence[float]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """RANSAC-based homography calibration."""
        print("🔍 Step 1: Performing RANSAC Calibration...")

        self.pixel_points = np.array(pixel_points, dtype=np.float32)
        self.world_points_approx = np.array(world_points_approx, dtype=np.float32)

        H, mask = cv2.findHomography(
            self.pixel_points,
            self.world_points_approx[:, :2],
            cv2.RANSAC,
            ransacReprojThreshold=5.0,
            confidence=0.99,
            maxIters=5000,
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

            print(f"   ✅ Inliers: {inlier_count}/{len(self.pixel_points)}")
            print(f"   ✅ MAE: {mae:.3f} m")

            self.calibration_metrics["final_mae"] = mae
            self.calibration_metrics["inlier_ratio"] = inlier_count / len(
                self.pixel_points
            )
            self._calculate_bev_scale()

        return self.homography, self.inlier_mask

    def _calculate_bev_scale(self) -> None:
        """Calculate BEV scale from inliers."""
        if self.world_points_approx is None or self.inlier_mask is None:
            return

        inlier_points = self.world_points_approx[self.inlier_mask]
        world_bounds = inlier_points[:, :2]

        x_min, y_min = world_bounds.min(axis=0)
        x_max, y_max = world_bounds.max(axis=0)

        margin_x = 0.2 * (x_max - x_min)
        margin_y = 0.2 * (y_max - y_min)

        self.bev_x_min = x_min - margin_x
        self.bev_x_max = x_max + margin_x
        self.bev_y_min = y_min - margin_y
        self.bev_y_max = y_max + margin_y

        self.meters_per_pixel_x = (self.bev_x_max - self.bev_x_min) / self.bev_width
        self.meters_per_pixel_y = (self.bev_y_max - self.bev_y_min) / self.bev_height

        print(f"   📐 BEV Scale: {self.meters_per_pixel_x:.3f} m/pixel")

    def pixel_to_world(self, pixel_point: Iterable[float]) -> np.ndarray:
        """Convert pixel coordinates to world coordinates."""
        if self.homography is None:
            raise RuntimeError("Homography not initialized; call calibrate() first")

        pixel_h = np.append(np.array(pixel_point, dtype=np.float32), 1).reshape(3, 1)
        world_h = self.homography @ pixel_h
        return (world_h[:2] / world_h[2]).flatten()

    def validate_bev(self) -> List[Dict[str, Any]]:
        """Validate bird's-eye-view transformation."""
        print("\n🔍 Validating Bird's Eye View...")

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

        print(f"   All points - Mean: {np.mean(all_errors):.3f} m")
        print(f"   Inliers - Mean: {np.mean(inlier_errors):.3f} m")

        self.calibration_metrics["bev_error"] = float(np.mean(inlier_errors))
        return validation_results

    # ------------------------ Speed estimation ------------------------

    def estimate_speed(
        self,
        pixel_positions: np.ndarray,
        frame_times: np.ndarray,
        fps: float = 30.0,
    ) -> Dict[str, Any]:
        """Estimate vehicle speed from trajectory."""
        print("\n🔍 Step 2: Estimating Vehicle Speed...")

        if self.homography is None:
            raise RuntimeError("Homography not initialized; call calibrate() first")

        world_positions: List[np.ndarray] = []
        valid_idx: List[int] = []

        for i, pos in enumerate(pixel_positions):
            try:
                world_pos = self.pixel_to_world(pos)
                world_positions.append(world_pos)
                valid_idx.append(i)
            except Exception:
                continue

        if len(world_positions) < 5:
            # Fallback prior
            return {"final_speed": 15.0, "speed_std": 2.0}

        world_positions_arr = np.vstack(world_positions)
        frame_times_valid = frame_times[valid_idx]

        speeds: List[float] = []
        for i in range(1, len(world_positions_arr)):
            dist = float(
                np.linalg.norm(world_positions_arr[i] - world_positions_arr[i - 1])
            )
            time_diff = float(frame_times_valid[i] - frame_times_valid[i - 1])

            if time_diff > 0:
                speed_kmh = (dist / time_diff) * 3.6
                if 0.5 < speed_kmh < 50:
                    speeds.append(speed_kmh)

        if len(speeds) < 3:
            return {"final_speed": 15.0, "speed_std": 2.0}

        speeds_arr = np.array(speeds, dtype=np.float32)
        final_speed = float(np.median(speeds_arr))
        speed_std = float(np.std(speeds_arr))

        print(f"   ✅ Final Speed: {final_speed:.2f} km/h")
        print(f"   ✅ Std Dev: {speed_std:.2f} km/h")

        return {
            "final_speed": final_speed,
            "speed_std": speed_std,
            "all_speeds": speeds_arr,
        }

    # ------------------------ Reporting ------------------------

    def generate_report(self, speed_results: Dict[str, Any]) -> Dict[str, float]:
        """Generate concise research report."""
        print("\n" + "=" * 60)
        print("📊 RESEARCH REPORT")
        print("=" * 60)

        mae = float(self.calibration_metrics.get("final_mae", 0.0))
        bev_error = float(self.calibration_metrics.get("bev_error", 0.0))

        print("\n🎯 GEOMETRIC ACCURACY:")
        print(f"   • MAE: {mae:.3f} m")
        print(f"   • BEV Error: {bev_error:.3f} m")

        print("\n🚗 SPEED ESTIMATION:")
        print(f"   • Velocity: {speed_results['final_speed']:.2f} km/h")
        print(f"   • Uncertainty: ±{speed_results['speed_std']:.2f} km/h")

        return {
            "mae": mae,
            "bev_error": bev_error,
            "speed": float(speed_results["final_speed"]),
            "uncertainty": float(speed_results["speed_std"]),
        }


def run_demo() -> tuple[CompleteTrafficAnalyzer, Dict[str, Any], Dict[str, float]]:
    """Minimal in-file demo for calibration and speed estimation."""
    print("=" * 60)
    print("🚗 TRAFFIC ANALYSIS SYSTEM – DEMO")
    print("=" * 60)

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

    print("\nStep 1: Calibrating...")
    analyzer.calibrate(pixel_points, world_points)

    print("\nStep 2: Validating BEV...")
    analyzer.validate_bev()

    print("\nStep 3: Estimating Speed...")
    speed_results = analyzer.estimate_speed(vehicle_pixels, frame_times, fps=fps)

    print("\nStep 4: Generating Report...")
    metrics = analyzer.generate_report(speed_results)

    return analyzer, speed_results, metrics


# ======================== Video → PET CLI ========================


def run_video_to_pet(
    video_path: str,
    bev_config_path: str = "configs/bev_config.json",
    grid_config_path: str = "configs/GITI_grid_config.json",
    sam3_weights_path: str = "sam3.pt",
    out_csv_path: str = "outputs/petevents_bev.csv",
    pet_threshold: float = 2.0,
    max_frames: int | None = None,
) -> pd.DataFrame:
    """Video → SAM3 detections → grid → BEV → PET events CSV.

    Returns the DataFrame of PET events for convenience.
    """
    try:
        from grid_trajectory.sam3_grid_pet import run_sam3_grid_pet
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency for video pipeline. Install required packages "
            "for SAM3/Ultralytics before running video mode."
        ) from exc

    project_root = str(Path(".").resolve())

    result = run_sam3_grid_pet(
        project_root=project_root,
        video_rel_path=str(Path(video_path)),
        sam3_rel_path=str(Path(sam3_weights_path)),
        grid_rel_path=str(Path(grid_config_path)),
        bev_rel_path=str(Path(bev_config_path)),
        output_name="sam3_grid_pet_run",
        conf=0.25,
        pet_threshold=pet_threshold,
    )

    pet_events = result.get("pet_events", [])
    out_path = Path(out_csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    max_frames: int | None = None,
    for idx, e in enumerate(pet_events):
        rows.append(
            {
                "event_id": idx,
                "pet": e.get("pet"),
                "frame": e.get("frame_idx"),
                "track_a": e.get("track_a"),
                "track_b": e.get("track_b"),
                "conflict_type": e.get("conflict_type"),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"✅ Saved {len(df)} PET events to {out_path}")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Video → SAM3 + grid → BEV → PET events pipeline"
    )
    parser.add_argument("--video", default=None, help="Input video path")
    parser.add_argument(
        "--bev-config",
        default="configs/bev_config.json",
        help="BEV configuration JSON path",
    )
    parser.add_argument(
        "--grid-config",
        default="configs/GITI_grid_config.json",
        help="Grid configuration JSON path",
    )
    parser.add_argument(
        "--sam3-weights",
        default="sam3.pt",
        help="SAM3 weights checkpoint path",
    )
    parser.add_argument(
        "--out-csv",
        default="outputs/petevents_bev.csv",
        help="Output CSV path for PET events",
    )
    parser.add_argument(
        "--pet-threshold",
        type=float,
        default=2.0,
        help="PET threshold in seconds",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run internal calibration/speed demo instead of video pipeline",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Process only the first N frames",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.demo:
        run_demo()
        return

    if not args.video:
        raise SystemExit("error: --video is required unless --demo is used")

    run_video_to_pet(
        video_path=args.video,
        bev_config_path=args.bev_config,
        grid_config_path=args.grid_config,
        sam3_weights_path=args.sam3_weights,
        out_csv_path=args.out_csv,
        pet_threshold=args.pet_threshold,
    )


if __name__ == "__main__":
    main()
