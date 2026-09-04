#!/usr/bin/env python
import argparse
import json
import logging
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

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
    """Research-oriented traffic analysis system with homography, BEV, and speed estimation."""

    def __init__(self, bev_width: int = 1000, bev_height: int = 800) -> None:
        self.homography: np.ndarray | None = None
        self.inv_homography: np.ndarray | None = None
        self.world_points_approx: np.ndarray | None = None
        self.pixel_points: np.ndarray | None = None
        self.inlier_mask: np.ndarray | None = None
        self.calibration_metrics: dict[str, float] = {}
        self.bev_width = bev_width
        self.bev_height = bev_height
        self.bev_x_min: float | None = None
        self.bev_x_max: float | None = None
        self.bev_y_min: float | None = None
        self.bev_y_max: float | None = None
        self.meters_per_pixel_x: float | None = None
        self.meters_per_pixel_y: float | None = None

    def calibrate(
        self,
        pixel_points: Sequence[Sequence[float]],
        world_points_approx: Sequence[Sequence[float]],
        ransac_threshold: float = 5.0,
        ransac_confidence: float = 0.99,
        ransac_max_iters: int = 5000,
    ) -> tuple[np.ndarray, np.ndarray | None]:
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
            errors = np.linalg.norm(projected - self.world_points_approx[:, :2], axis=1)
            mae = float(np.mean(errors[self.inlier_mask]))
            self.calibration_metrics["final_mae"] = mae
            self.calibration_metrics["inlier_ratio"] = inlier_count / len(self.pixel_points)
            self._calculate_bev_scale()

        return self.homography, self.inlier_mask

    def _calculate_bev_scale(self, safety_margin: float = 0.2) -> None:
        if self.world_points_approx is None or self.inlier_mask is None:
            return
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

    def pixel_to_world(self, pixel_point: Iterable[float]) -> np.ndarray:
        if self.homography is None:
            raise RuntimeError("Homography not initialized; call calibrate() first")
        pixel_h = np.append(np.array(pixel_point, dtype=np.float32), 1.0)
        world_h = self.homography @ pixel_h
        return world_h[:2] / world_h[2]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NNDS traffic analyzer demo")
    p.add_argument("--self-test", action="store_true", help="Run a tiny calibration self-test")
    p.add_argument("--save-json", type=str, default="", help="Write self-test output to JSON")
    return p


def run_self_test() -> dict:
    analyzer = CompleteTrafficAnalyzer()
    pixel_points = [(0, 0), (10, 0), (10, 10), (0, 10)]
    world_points = [(0, 0, 0), (5, 0, 0), (5, 5, 0), (0, 5, 0)]
    H, mask = analyzer.calibrate(pixel_points, world_points)
    return {
        "version": __version__,
        "homography_shape": list(H.shape),
        "inliers": int(mask.sum()) if mask is not None else None,
        "metrics": analyzer.calibration_metrics,
    }


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = build_parser().parse_args(argv)
    if args.self_test:
        result = run_self_test()
        print(json.dumps(result, indent=2))
        if args.save_json:
            Path(args.save_json).write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
