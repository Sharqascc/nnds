import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from datetime import timedelta

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

    We count a crossing whenever a tracked object's anchor point moves from
    one side of the infinite line to the other.

    Direction convention:
    - If entry_side == "left":
        negative -> positive  => entry
        positive -> negative  => exit
    - If entry_side == "right":
        negative -> positive  => exit
        positive -> negative  => entry
    """
    name: str
    p1: Tuple[int, int]
    p2: Tuple[int, int]
    color: Tuple[int, int, int] = (0, 255, 255)   # BGR for OpenCV drawing
    entry_side: str = "left"

    entry_count: int = 0
    exit_count: int = 0

    min_frames_between_crossings: int = 10
    history: Dict[int, int] = field(default_factory=dict)

    def direction(self) -> np.ndarray:
        v = np.array(self.p2, dtype=float) - np.array(self.p1, dtype=float)
        n = np.linalg.norm(v)
        if n == 0:
            return np.array([1.0, 0.0], dtype=float)
        return v / n

    def signed_distance(self, point: Tuple[float, float]) -> float:
        """
        Signed side value of a point relative to the line through p1->p2.
        Only the sign matters for crossing detection.
        """
        p = np.array(point, dtype=float)
        a = np.array(self.p1, dtype=float)
        d = self.direction()
        ap = p - a
        return float(ap[0] * d[1] - ap[1] * d[0])

    def check_crossing(
        self,
        prev_pos: Optional[Tuple[float, float]],
        curr_pos: Optional[Tuple[float, float]],
        track_id: int,
        frame_idx: int,
    ) -> Optional[str]:
        """
        Returns:
            "entry", "exit", or None
        """
        if prev_pos is None or curr_pos is None:
            return None

        prev_side = self.signed_distance(prev_pos)
        curr_side = self.signed_distance(curr_pos)

        if prev_side == 0 or curr_side == 0:
            return None

        if prev_side * curr_side >= 0:
            return None

        last_frame = self.history.get(track_id, -10**9)
        if frame_idx - last_frame < self.min_frames_between_crossings:
            return None

        self.history[track_id] = frame_idx

        neg_to_pos = (prev_side < 0 and curr_side > 0)

        if self.entry_side.lower() == "left":
            if neg_to_pos:
                self.entry_count += 1
                return "entry"
            else:
                self.exit_count += 1
                return "exit"
        else:  # entry_side == "right"
            if neg_to_pos:
                self.exit_count += 1
                return "exit"
            else:
                self.entry_count += 1
                return "entry"


# -----------------------------
# Lightweight tracker wrapper
# -----------------------------
@dataclass
class RobustTracker:
    """
    Expects each detection to optionally carry a stable 'track_id' from an
    external tracker (e.g. ByteTrack/OC-SORT).

    If 'track_id' is missing, assigns a new ID.
    Keeps tracks alive for up to `max_missing` frames.
    """
    max_missing: int = 30
    next_id: int = 1
    tracks: Dict[int, Dict] = field(default_factory=dict)

    def update(self, detections: List[Dict]) -> Dict[int, Dict]:
        updated_tracks: Dict[int, Dict] = {}

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

    detector(frame) must return list[dict] with at least:
      - "centroid": (x, y)
      - optional "track_id"
      - optional "cls"
      - optional "conf"
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
                "auto rickshaw",
                "rickshaw",
                "auto-rickshaw",
            ]

        self.gates: Dict[str, VirtualGate] = {}
        if self.gate_config is not None:
            self.gates = self.load_gates(self.gate_config)

        self.tracker = RobustTracker()
        self.last_event: Optional[str] = None

    # ---------- gate config ----------
    @staticmethod
    def load_gates(configfile: str) -> Dict[str, VirtualGate]:
        """
        Expected YAML:
        gates:
          - name: "West_Gate"
            start: [210, 170]
            end: [450, 150]
            color: [255, 0, 0]      # RGB in YAML
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

            rgb = g.get("color", [255, 255, 0])
            if isinstance(rgb, list) and len(rgb) == 3:
                color = (int(rgb[2]), int(rgb[1]), int(rgb[0]))  # RGB -> BGR
            else:
                color = (0, 255, 255)

            entry_side = str(g.get("entry_side", "left")).lower().strip()
            if entry_side not in {"left", "right"}:
                entry_side = "left"

            gates[name] = VirtualGate(
                name=name,
                p1=(int(start[0]), int(start[1])),
                p2=(int(end[0]), int(end[1])),
                color=color,
                entry_side=entry_side,
            )

        logger.info("Loaded %d virtual gates from %s", len(gates), configfile)
        return gates

    # ---------- helpers ----------
    @staticmethod
    def _format_time(frame_idx: int, fps: float) -> str:
        if fps <= 0:
            return "00:00:00"
        seconds = int(frame_idx / fps)
        return str(timedelta(seconds=seconds))

    def _compute_totals(self) -> Tuple[int, int]:
        total_in = sum(g.entry_count for g in self.gates.values())
        total_out = sum(g.exit_count for g in self.gates.values())
        return total_in, total_out

    def _normalize_class_name(self, cls_name: Optional[str]) -> str:
        if cls_name is None:
            return "object"
        return str(cls_name).strip().lower()

    def _allowed_detection(self, det: Dict) -> bool:
        conf = float(det.get("conf", 1.0))
        if conf < self.min_confidence:
            return False

        cls_name = det.get("cls", det.get("class_name", "object"))
        cls_name = self._normalize_class_name(cls_name)

        if self.classes_of_interest and cls_name not in {
            self._normalize_class_name(c) for c in self.classes_of_interest
        }:
            return False

        centroid = det.get("centroid")
        if centroid is None or len(centroid) != 2:
            return False

        return True

    def _draw_gate_labels(self, vis: np.ndarray) -> None:
        for gate in self.gates.values():
            cv2.line(vis, gate.p1, gate.p2, gate.color, 2)

            mx = int((gate.p1[0] + gate.p2[0]) / 2)
            my = int((gate.p1[1] + gate.p2[1]) / 2)

            label = f"{gate.name}"
            cv2.putText(
                vis,
                label,
                (mx, my),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                gate.color,
                2,
                cv2.LINE_AA,
            )

    def _draw_tracks(self, vis: np.ndarray, tracks: Dict[int, Dict]) -> None:
        for tid, t in tracks.items():
            c = t.get("centroid")
            if c is None:
                continue

            x, y = int(c[0]), int(c[1])
            cls_name = t.get("cls", "object")
            conf = float(t.get("conf", 1.0))

            cv2.circle(vis, (x, y), 4, (0, 0, 255), -1)

            label = f"ID {tid} {cls_name} {conf:.2f}"
            cv2.putText(
                vis,
                label,
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def _draw_stats_panel(self, frame: np.ndarray, frame_idx: int, fps: float) -> np.ndarray:
        h, w = frame.shape[:2]

        total_in, total_out = self._compute_totals()
        lines: List[str] = [
            f"Frame: {frame_idx}",
            f"Time: {self._format_time(frame_idx, fps)}",
            f"TOTAL ENTRY: {total_in}",
            f"TOTAL EXIT : {total_out}",
            ""
        ]

        for gate in self.gates.values():
            lines.append(
                f"{gate.name}: IN {gate.entry_count} | OUT {gate.exit_count}"
            )

        if self.last_event:
            lines.append("")
            lines.append(f"LAST EVENT: {self.last_event}")

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.62
        thickness = 2
        line_h = 25
        pad = 12

        text_widths = []
        for text in lines:
            if text == "":
                continue
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_widths.append(tw)

        panel_w = min(max(text_widths, default=260) + 36, w - 20)
        panel_h = min(line_h * len(lines) + 2 * pad, h - 20)

        x1, y1 = 10, 10
        x2, y2 = x1 + panel_w, y1 + panel_h

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
        frame = cv2.addWeighted(overlay, 0.60, frame, 0.40, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 220, 220), 2)

        y = y1 + pad + 10
        for text in lines:
            if text == "":
                y += line_h // 2
                continue

            color = (255, 255, 255)
            x_text = x1 + 12

            if text.startswith("TOTAL ENTRY"):
                color = (80, 255, 80)
            elif text.startswith("TOTAL EXIT"):
                color = (80, 200, 255)
            elif ": IN " in text and "| OUT " in text:
                gate_name = text.split(":")[0]
                gate = self.gates.get(gate_name)
                if gate is not None:
                    cv2.rectangle(frame, (x1 + 10, y - 12), (x1 + 20, y - 2), gate.color, -1)
                    x_text = x1 + 28
            elif text.startswith("LAST EVENT"):
                color = (0, 255, 255)

            cv2.putText(
                frame,
                text,
                (x_text, y),
                font,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )
            y += line_h

        return frame

    # ---------- main processing ----------
    def process_video(
        self,
        detector: Callable[[np.ndarray], List[Dict]],
        output_video: Optional[str] = None,
        max_frames: Optional[int] = None,
        log_visual_debug: bool = False,
    ) -> Dict:
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
        self.last_event = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if max_frames is not None and frame_idx > max_frames:
                break

            raw_detections = detector(frame)
            detections: List[Dict] = []

            for d in raw_detections:
                if not self._allowed_detection(d):
                    continue

                d = dict(d)
                d["frame_idx"] = frame_idx
                d["cls"] = self._normalize_class_name(d.get("cls", d.get("class_name", "object")))
                d["conf"] = float(d.get("conf", 1.0))
                detections.append(d)

            tracks = self.tracker.update(detections)

            for tid, t in tracks.items():
                curr = t.get("centroid")
                prev = t.get("prev_centroid")

                for gate in self.gates.values():
                    status = gate.check_crossing(prev, curr, tid, frame_idx)
                    if status is not None:
                        event = f"{gate.name} {status.upper()} | ID {tid} | {t.get('cls', 'object')}"
                        self.last_event = event
                        logger.info("Frame %d: %s", frame_idx, event)

                t["prev_centroid"] = curr

            if out is not None:
                vis = frame.copy()

                self._draw_gate_labels(vis)

                if log_visual_debug:
                    self._draw_tracks(vis, tracks)

                vis = self._draw_stats_panel(vis, frame_idx=frame_idx, fps=fps)

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
