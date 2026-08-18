#!/usr/bin/env python
"""Gate Counter - Traffic Volume Estimation Module (Stub)

Note: Full implementation moved to parent directory for now.
This stub allows imports from core module.

Usage:
    from core import TrafficVolumeCounter, VirtualGate
"""

from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional, Any
import numpy as np


@dataclass
class VirtualGate:
    """A virtual counting gate defined by a line segment between p1 and p2."""
    name: str
    p1: Tuple[int, int]
    p2: Tuple[int, int]
    color: Tuple[int, int, int] = (0, 255, 255)
    entry_side: str = "left"
    entry_count: int = 0
    exit_count: int = 0

    def signed_distance(self, point: Tuple[float, float]) -> float:
        """Signed side value of a point relative to the line through p1->p2."""
        p = np.array(point, dtype=float)
        a = np.array(self.p1, dtype=float)
        d = self.direction()
        ap = p - a
        return float(ap[0] * d[1] - ap[1] * d[0])

    def direction(self) -> np.ndarray:
        """Normalized direction vector from p1 to p2."""
        v = np.array(self.p2, dtype=float) - np.array(self.p1, dtype=float)
        n = np.linalg.norm(v)
        if n == 0:
            return np.array([1.0, 0.0], dtype=float)
        return v / n


class TrafficVolumeCounter:
    """High-level driver for gate-based traffic volume estimation."""

    def __init__(
        self,
        videopath: str,
        gate_config: Optional[str] = None,
        classes_of_interest: Optional[List[str]] = None,
        min_confidence: float = 0.25,
    ):
        self.videopath = videopath
        self.gate_config = gate_config
        self.classes_of_interest = classes_of_interest or ["car", "motorcycle", "bus"]
        self.min_confidence = min_confidence
        self.gates: Dict[str, VirtualGate] = {}

    def process_video(
        self,
        detector,
        output_video: Optional[str] = None,
        max_frames: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Process video with detector and return traffic counts."""
        return {
            "total_entries": 0,
            "total_exits": 0,
            "gates": {},
        }


__all__ = [
    "TrafficVolumeCounter",
    "VirtualGate",
]
