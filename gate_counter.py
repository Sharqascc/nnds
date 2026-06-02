GATE_HYSTERESIS = {
    "West_Gate": {"min_side_mag": 0.10, "min_delta_side": 0.15},
    "North_Gate": {"min_side_mag": 0.35, "min_delta_side": 0.50},
    "SouthWest_Gate": {"min_side_mag": 0.50, "min_delta_side": 0.75},
    "South_Gate": {"min_side_mag": 0.35, "min_delta_side": 0.50},
    "East_Gate": {"min_side_mag": 0.35, "min_delta_side": 0.50},
}

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import pandas as pd
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

    def __post_init__(self) -> None:
        if self.min_frames_between_crossings < 1:
            self.min_frames_between_crossings = 1

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

        # Treat near-zero side values as no reliable crossing
        eps = 1e-6
        if abs(prev_side) < eps or abs(curr_side) < eps:
            return None

        if prev_side * curr_side >= 0:
            return None

        cfg = GATE_HYSTERESIS.get(
            self.name,
            {"min_side_mag": 0.35, "min_delta_side": 0.50},
        )
        min_side_mag = cfg["min_side_mag"]
        min_delta_side = cfg["min_delta_side"]

        if abs(prev_side) < min_side_mag or abs(curr_side) < min_side_mag:
            return None

        if abs(curr_side - prev_side) < min_delta_side:
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
    Keeps tracks alive for up to `max_missing` frames and can drop very old tracks.
    """
    max_missing: int = 30
    max_track_age_frames: int = 300
    next_id: int = 1
    tracks: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    _frame_counter: int = 0

    def update(self, detections: List[Dict[str, Any]], frame_idx: int) -> Dict[int, Dict[str, Any]]:
        updated_tracks: Dict[int, Dict[str, Any]] = {}
        remaining_track_ids = set(self.tracks.keys())

        def _dist(a, b):
            return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)

        for det in detections:
            centroid = det.get("centroid")
            tid = det.get("track_id")

            if tid is None and centroid is not None:
                best_tid = None
                best_dist = float("inf")

                for existing_tid in list(remaining_track_ids):
                    prev = self.tracks.get(existing_tid, {}).get("centroid")
                    if prev is None:
                        continue
                    d = _dist(centroid, prev)
                    if d < best_dist and d <= 75.0:
                        best_dist = d
                        best_tid = existing_tid

                if best_tid is not None:
                    tid = best_tid
                    det["track_id"] = tid
                    remaining_track_ids.discard(tid)

            if tid is None:
                tid = self.next_id
                self.next_id += 1
                det["track_id"] = tid

            if tid in self.tracks and "centroid" in self.tracks[tid]:
                det["prev_centroid"] = self.tracks[tid].get("centroid")
            elif tid in self.tracks and "prev_centroid" in self.tracks[tid]:
                det["prev_centroid"] = self.tracks[tid].get("prev_centroid")

            det["missed"] = 0
            det["last_seen_frame"] = frame_idx
            updated_tracks[tid] = det
            remaining_track_ids.discard(tid)

        for tid, old in self.tracks.items():
            if tid in updated_tracks:
                continue

            missed = int(old.get("missed", 0)) + 1
            age = frame_idx - int(old.get("last_seen_frame", frame_idx))

            if missed <= self.max_missing and age <= self.max_track_age_frames:
                keep = dict(old)
                keep["missed"] = missed
                updated_tracks[tid] = keep

        self.tracks = updated_tracks
        return self.tracks

    def _cleanup_old_tracks(self) -> None:
        to_delete: List[int] = []
        for tid, t in self.tracks.items():
            last_seen = t.get("last_seen_frame", 0)
            if self._frame_counter - last_seen > self.max_track_age_frames:
                to_delete.append(tid)

        for tid in to_delete:
            self.tracks.pop(tid, None)


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

    Example detector that returns only cars:
        def car_detector(frame):
            detections = model(frame)
            return [d for d in detections if d.get('cls') == 'car']
    """
    videopath: str
    gate_config: Optional[str] = None
    classes_of_interest: Optional[List[str]] = None
    min_confidence: float = 0.25

    # drawing / performance flags
    draw_stats: bool = True
    draw_tracks: bool = False

    def __post_init__(self) -> None:
        if not Path(self.videopath).exists():
            raise FileNotFoundError(f"Video not found: {self.videopath}")

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

        # normalized class whitelist for fast membership checks
        self._class_whitelist = {
            self._normalize_class_name(c) for c in self.classes_of_interest
        }

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
            enabled: true
        """
        path = Path(configfile)
        gates: Dict[str, VirtualGate] = {}

        if not path.exists():
            logger.warning(
                "Gate config file not found at %s; no gates will be used.",
                configfile,
            )
            return gates

        try:
            with path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            logger.error("Failed to parse gate config %s: %s", configfile, e)
            return gates

        items = cfg.get("gates", [])
        for g in items:
            if g.get("enabled", True) is False:
                continue

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

    def _allowed_detection(self, det: Dict[str, Any]) -> bool:
        conf = float(det.get("conf", 1.0))
        if conf < self.min_confidence:
            return False

        cls_name = det.get("cls", det.get("class_name", "object"))
        cls_name = self._normalize_class_name(cls_name)

        if self._class_whitelist and cls_name not in self._class_whitelist:
            return False

        centroid = det.get("centroid")
        if centroid is None or len(centroid) != 2:
            bbox = det.get("bbox")
            if bbox is None or len(bbox) != 4:
                return False
            x1, y1, x2, y2 = bbox
            centroid = ((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0)
            det["centroid"] = centroid

        return True

    def _draw_gate_labels(self, vis: np.ndarray) -> None:
        h, w = vis.shape[:2]
        for gate in self.gates.values():
            # clip gate endpoints to frame bounds
            p1 = (max(0, min(w - 1, gate.p1[0])), max(0, min(h - 1, gate.p1[1])))
            p2 = (max(0, min(w - 1, gate.p2[0])), max(0, min(h - 1, gate.p2[1])))

            cv2.line(vis, p1, p2, gate.color, 2)

            mx = int((p1[0] + p2[0]) / 2)
            my = int((p1[1] + p2[1]) / 2)

            # Keep label inside frame
            mx = max(10, min(w - 10, mx))
            my = max(10, min(h - 10, my))

            cv2.putText(
                vis,
                gate.name,
                (mx, my),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                gate.color,
                2,
                cv2.LINE_AA,
            )

    def _draw_tracks(self, vis: np.ndarray, tracks: Dict[int, Dict[str, Any]]) -> None:
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

    def _draw_stats_panel(
        self, frame: np.ndarray, frame_idx: int, fps: float
    ) -> np.ndarray:
        h, w = frame.shape[:2]

        total_in, total_out = self._compute_totals()
        lines: List[str] = [
            f"Frame: {frame_idx}",
            f"Time: {self._format_time(frame_idx, fps)}",
            f"TOTAL ENTRY: {total_in}",
            f"TOTAL EXIT : {total_out}",
            "",
        ]

        for gate in self.gates.values():
            lines.append(f"{gate.name}: IN {gate.entry_count} | OUT {gate.exit_count}")

        if self.last_event:
            lines.append("")
            lines.append(f"LAST EVENT: {self.last_event}")

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        line_h = 18
        pad = 10

        text_widths = []
        for text in lines:
            if text == "":
                continue
            (tw, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_widths.append(tw)

        panel_w = min(max(text_widths, default=220) + 26, w - 20)
        panel_h = min(line_h * len(lines) + 2 * pad, h - 20)

        x2, y2 = w - 10, h - 10
        x1, y1 = x2 - panel_w, y2 - panel_h

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
        frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 220, 220), 1)

        y = y1 + pad + 6
        for text in lines:
            if text == "":
                y += line_h // 2
                continue

            color = (255, 255, 255)
            x_text = x1 + 10

            if text.startswith("TOTAL ENTRY"):
                color = (80, 255, 80)
            elif text.startswith("TOTAL EXIT"):
                color = (80, 200, 255)
            elif ": IN " in text and "| OUT " in text:
                gate_name = text.split(":")[0]
                gate = self.gates.get(gate_name)
                if gate is not None:
                    cv2.rectangle(
                        frame,
                        (x1 + 8, y - 10),
                        (x1 + 16, y - 2),
                        gate.color,
                        -1,
                    )
                    x_text = x1 + 22
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


    def _normalize_detector_output(
        self,
        raw_output: Any,
    ) -> List[Dict[str, Any]]:
        """
        Convert detector outputs into a standard list-of-dicts format.

        Supported inputs:
        - list[dict]
        - tuple/list containing Ultralytics Results
        - single Ultralytics Results object
        """
        if raw_output is None:
            return []

        if isinstance(raw_output, list):
            if not raw_output:
                return []
            if all(isinstance(x, dict) for x in raw_output):
                return raw_output
            if len(raw_output) == 1:
                raw_output = raw_output[0]

        if isinstance(raw_output, dict):
            return [raw_output]

        if hasattr(raw_output, "boxes"):
            result = raw_output
            names = getattr(result, "names", {}) or {}
            detections: List[Dict[str, Any]] = []

            if result.boxes is None:
                return detections

            for box in result.boxes:
                cls_id = int(box.cls[0].item()) if getattr(box, "cls", None) is not None else -1
                conf = float(box.conf[0].item()) if getattr(box, "conf", None) is not None else 0.0
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append(
                    {
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "class_id": cls_id,
                        "class_name": names.get(cls_id, str(cls_id)),
                        "cls": names.get(cls_id, str(cls_id)),
                        "conf": conf,
                        "track_id": None,
                    }
                )
            return detections

        raise TypeError(
            f"Unsupported detector output type: {type(raw_output)!r}. "
            "Expected list[dict], dict, or Ultralytics Results."
        )


    # ---------- main processing ----------
    def process_video(
        self,
        detector: Callable[[np.ndarray], List[Dict[str, Any]]],
        output_video: Optional[str] = None,
        max_frames: Optional[int] = None,
        log_visual_debug: bool = False,
        show_progress: bool = False,
    ) -> Dict[str, Any]:
        cap: Optional[cv2.VideoCapture] = None
        out: Optional[cv2.VideoWriter] = None
        pbar = None

        try:
            cap = cv2.VideoCapture(self.videopath)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {self.videopath}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 1e-3:
                logger.warning(
                    "FPS not found or invalid in video metadata, defaulting to 25.0"
                )
                fps = 25.0

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if output_video is not None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if max_frames is not None:
                total_frames = min(total_frames, max_frames)

            if show_progress:
                try:
                    from tqdm.auto import tqdm
                    pbar = tqdm(total=total_frames, desc="Gate counting")
                except ImportError:
                    pbar = None

            frame_idx = 0
            self.last_event = None

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if max_frames is not None and frame_idx > max_frames:
                    break

                raw_output = detector(frame)
                raw_detections = self._normalize_detector_output(raw_output)
                detections: List[Dict[str, Any]] = []

                for d in raw_detections:
                    if not self._allowed_detection(d):
                        continue

                    d = dict(d)
                    d["frame_idx"] = frame_idx
                    d["cls"] = self._normalize_class_name(
                        d.get("cls", d.get("class_name", "object"))
                    )
                    d["conf"] = float(d.get("conf", 1.0))
                    detections.append(d)

                tracks = self.tracker.update(detections, frame_idx=frame_idx)

                for tid, t in tracks.items():
                    curr = t.get("centroid")
                    prev = t.get("prev_centroid")

                    for gate in self.gates.values():
                        status = gate.check_crossing(prev, curr, tid, frame_idx)
                        if status is not None:
                            event = (
                                f"{gate.name} {status.upper()} | ID {tid} | "
                                f"{t.get('cls', 'object')}"
                            )
                            self.last_event = event
                            logger.info("Frame %d: %s", frame_idx, event)

                    t["prev_centroid"] = curr

                if out is not None:
                    vis = frame.copy()

                    self._draw_gate_labels(vis)

                    if log_visual_debug or self.draw_tracks:
                        self._draw_tracks(vis, tracks)

                    if self.draw_stats:
                        vis = self._draw_stats_panel(vis, frame_idx=frame_idx, fps=fps)

                    out.write(vis)

                if pbar is not None:
                    pbar.update(1)

            total_in, total_out = self._compute_totals()
            result: Dict[str, Any] = {
                "total_entries": total_in,
                "total_exits": total_out,
                "gates": {
                    name: {
                        "entries": g.entry_count,
                        "exits": g.exit_count,
                    }
                    for name, g in self.gates.items()
                },
            }

            logger.info(
                "Traffic volume result: total_entries=%d total_exits=%d",
                result["total_entries"],
                result["total_exits"],
            )
            logger.debug("Traffic volume result detail: %s", result)

            return result

        finally:
            if cap is not None:
                cap.release()
            if out is not None:
                out.release()
            if pbar is not None:
                pbar.close()

    # ---------- results export ----------
    @staticmethod
    def save_results(result: Dict[str, Any], path: str | Path) -> None:
        """Save per-gate counts to CSV for downstream analysis."""
        path = Path(path)
        rows = []
        for name, stats in result.get("gates", {}).items():
            rows.append(
                {
                    "gate": name,
                    "entries": stats.get("entries", 0),
                    "exits": stats.get("exits", 0),
                }
            )
        df = pd.DataFrame(rows)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
