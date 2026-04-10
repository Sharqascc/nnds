import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)


# -----------------------------
# Virtual gate representation
# -----------------------------

@dataclass
class VirtualGate:
    """
    A virtual counting gate defined by a line segment between p1 and p2
    in image pixel coordinates.

    We count a crossing whenever a tracked object's centroid moves from one
    side of the infinite line to the other.
    """
    name: str
    p1: Tuple[int, int]
    p2: Tuple[int, int]

    entry_count: int = 0
    exit_count: int = 0

    # To avoid double-counting noisy back-and-forth, require a gap in frames
    min_frames_between_crossings: int = 10
    # track_id -> last frame where a crossing was counted
    history: Dict[int, int] = field(default_factory=dict)

    def direction(self) -> np.ndarray:
        """
        Unit direction vector from p1 to p2.
        """
        v = np.array(self.p2, dtype=float) - np.array(self.p1, dtype=float)
        n = np.linalg.norm(v)
        if n == 0:
            # Degenerate; default to x-axis
            return np.array([1.0, 0.0], dtype=float)
        return v / n

    def signed_distance(self, point: Tuple[float, float]) -> float:
        """
        Signed distance of `point` from the gate line.

        Positive/negative sign indicates which side of the line the point is on.
        """
        p = np.array(point, dtype=float)
        a = np.array(self.p1, dtype=float)
        d = self.direction()
        ap = p - a

        # 2D cross product (scalar) between direction and AP
        # This is proportional to signed distance to the infinite line:
        #   cross(d, ap) = d.x * ap.y - d.y * ap.x
        # We use a consistent sign only; magnitude units are arbitrary.
        return float(ap[0] * d[1] - ap[1] * d[0])

    def check_crossing(
        self,
        prev_pos: Optional[Tuple[float, float]],
        curr_pos: Optional[Tuple[float, float]],
        track_id: int,
        frame_idx: int,
    ) -> Optional[str]:
        """
        Check whether a single tracked object has crossed this gate
        between its previous and current centroid positions.

        Returns:
            "entry", "exit", or None if no crossing was counted.
        """
        if prev_pos is None or curr_pos is None:
            return None

        prev_side = self.signed_distance(prev_pos)
        curr_side = self.signed_distance(curr_pos)

        # If either position lies exactly on the line or both are on the same side,
        # we do not count a crossing.
        if prev_side == 0 or curr_side == 0:
            return None
        if prev_side * curr_side >= 0:
            return None

        # Side changed: candidate crossing.
        # Debounce per track_id.
        last_frame = self.history.get(track_id, -10**9)
        if frame_idx - last_frame < self.min_frames_between_crossings:
            return None

        self.history[track_id] = frame_idx

        # Define "entry" vs "exit" by sign convention:
        # prev_side < 0 and curr_side > 0 -> one direction,
        # else -> the opposite direction.
        if prev_side < 0 and curr_side > 0:
            self.entry_count += 1
            return "entry"
        else:
            self.exit_count += 1
            return "exit"


# -----------------------------
# Lightweight tracker wrapper
# -----------------------------

@dataclass
class RobustTracker:
    """
    Very simple track manager that expects each detection to optionally carry
    a stable 'track_id' from an external tracker (e.g. ByteTrack/OC-SORT).

    If 'track_id' is missing, it assigns a new ID and keeps tracks alive for
    up to `max_missing` frames to allow gate logic to see short occlusions.
    """
    max_missing: int = 30
    next_id: int = 1
    tracks: Dict[int, Dict] = field(default_factory=dict)

    def update(self, detections: List[Dict]) -> Dict[int, Dict]:
        """
        Update internal track dictionary with current-frame detections.

        Each detection should at least contain:
            - "centroid": (x, y) in pixels
            - optionally: "track_id" from an upstream tracker
        """
        updated_tracks: Dict[int, Dict] = {}

        # First, incorporate current detections
        for det in detections:
            tid = det.get("track_id")
            if tid is None:
                tid = self.next_id
                self.next_id += 1
                det["track_id"] = tid

            if tid in self.tracks and "centroid" in self.tracks[tid]:
                det["prev_centroid"] = self.tracks[tid].get("centroid")
            elif tid in self.tracks and "prev_centroid" in self.tracks[tid]:
                det["prev_centroid"] = self.tracks[tid].get("prev_centroid")

            det["missed"] = 0
            updated_tracks[tid] = det

        # Carry-over old tracks that were not detected in this frame
        for tid, t in self.tracks.items():
            if tid not in updated_tracks:
                missed = t.get("missed", 0) + 1
                if missed <= self.max_missing:
                    t["missed"] = missed
                    updated_tracks[tid] = t

        self.tracks = updated_tracks
        return self.tracks


# -----------------------------
# Traffic volume counter
# -----------------------------

@dataclass
class TrafficVolumeCounter:
    """
    High-level driver for gate-based traffic volume estimation.

    You provide:
      - video path
      - YAML gate configuration (configs/gate_config.yaml)
      - a detector function that maps a frame -> list[dict] with centroids
    """
    videopath: str
    gate_config: Optional[str] = None
    classes_of_interest: Optional[List[str]] = None
    min_confidence: float = 0.25

    def __post_init__(self) -> None:
        if self.classes_of_interest is None:
            self.classes_of_interest = [
                "car",
                "motorcycle",
                "bus",
                "truck",
                "bicycle",
                "person",
            ]

        self.gates: Dict[str, VirtualGate] = {}
        if self.gate_config is not None:
            self.gates = self.load_gates(self.gate_config)

        self.tracker = RobustTracker()

    # ---------- gate config ----------

    @staticmethod
    def load_gates(configfile: str) -> Dict[str, VirtualGate]:
        """
        Load virtual gates from a YAML file with a top-level `gates` list.

        Expected YAML structure (see configs/gate_config.yaml):

        gates:
          - name: "West_Gate"
            start: [210, 170]
            end: [450, 150]
            color: [255, 0, 0]
            entry_side: "right"
        """
        path = Path(configfile)
        gates: Dict[str, VirtualGate] = {}

        if not path.exists():
            logger.warning("No gate config found at %s", configfile)
            return gates

        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        items = cfg.get("gates", [])
        for g in items:
            name = g.get("name")
            if not name:
                continue
            start = tuple(g.get("start", [0, 0]))
            end = tuple(g.get("end", [0, 0]))
            gates[name] = VirtualGate(name=name, p1=start, p2=end)

        logger.info("Loaded %d virtual gates from %s", len(gates), configfile)
        return gates

    # ---------- main processing ----------

    def process_video(
        self,
        detector: Callable[[np.ndarray], List[Dict]],
        output_video: Optional[str] = None,
        max_frames: Optional[int] = None,
        log_visual_debug: bool = False,
    ) -> Dict:
        """
        Run the volume counting over the video using the provided detector.

        detector(frame) must return a list of dicts with at least:
            - "centroid": (x, y) in pixels (float or int)
            - optional "track_id": int (stable across frames)
            - optional "cls" / "class_name": string
            - optional "conf": float
        """
        cap = cv2.VideoCapture(self.videopath)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.videopath}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if output_video is not None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))
        else:
            out = None

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            if max_frames is not None and frame_idx > max_frames:
                break

            # Get raw detections from user-supplied detector
            detections = detector(frame)

            # Stamp frame index
            for d in detections:
                d["frame_idx"] = frame_idx

            # Update track states
            tracks = self.tracker.update(detections)

            # Gate crossing logic
            for tid, t in tracks.items():
                curr = t.get("centroid")
                prev = t.get("prev_centroid")

                for gate in self.gates.values():
                    status = gate.check_crossing(prev, curr, tid, frame_idx)
                    if status is not None:
                        logger.info(
                            "Frame %d: track %d %s %s",
                            frame_idx,
                            tid,
                            status,
                            gate.name,
                        )

                # Update state for next frame
                t["prev_centroid"] = curr

            # Optional visualization
            if out is not None:
                vis = frame.copy()

                # Draw gates
                for gate in self.gates.values():
                    cv2.line(vis, gate.p1, gate.p2, (0, 255, 255), 2)
                    cv2.putText(
                        vis,
                        gate.name,
                        (int((gate.p1[0] + gate.p2[0]) / 2), int((gate.p1[1] + gate.p2[1]) / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                if log_visual_debug:
                    for t in tracks.values():
                        c = t.get("centroid")
                        if c is not None:
                            cv2.circle(
                                vis,
                                (int(c[0]), int(c[1])),
                                3,
                                (0, 0, 255),
                                -1,
                            )

                out.write(vis)

        cap.release()
        if out is not None:
            out.release()

        result = {
            "gates": {
                name: {
                    "entries": g.entry_count,
                    "exits": g.exit_count,
                }
                for name, g in self.gates.items()
            }
        }
        logger.info("Traffic volume result: %s", result)
        return result