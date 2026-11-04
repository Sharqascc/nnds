"""Render an annotated video from a completed pipeline run.

Post-processing step: given the original video, the events CSV, and
(optionally) the per-frame detections CSV that the pipeline wrote
alongside it, produce an annotated MP4 that shows:

- Original video frames
- Bounding boxes with track IDs (from detections CSV, if present)
- Grid cell boundaries (from grid config JSON, if present)
- Gate positions (from gate config YAML, if present)
- Conflict annotations (PET value + type) at the frame of each event

This module does not modify the pipeline. It reads CSVs from disk, so it
works the same for every backend that produces those files. If a
detections CSV is unavailable (some backends do not write one), boxes
are skipped and grid/gates/event annotations still render.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

try:
    from src.analysis.visualization.video_overlays import COLORS_BGR as _C

    _PALETTE = (
        list(_C.values()) if _C else [(0, 114, 178), (230, 159, 0), (0, 158, 115), (213, 94, 0)]
    )
except ImportError:
    _PALETTE = [(0, 114, 178), (230, 159, 0), (0, 158, 115), (213, 94, 0)]
if not _PALETTE:
    _PALETTE = [(0, 114, 178)]


def _load_detections(csv_path: Path) -> dict[int, list[dict[str, Any]]]:
    """Return {frame: [{track_id, bbox, class}, ...]}. Missing file -> {}."""
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    out: dict[int, list[dict[str, Any]]] = {}
    for _, row in df.iterrows():
        fval = row.get("frame", row.get("Frame", None))
        if pd.isna(fval):
            continue
        frame = int(fval)
        bbox = (
            int(row.get("x1", row.get("X1", 0))),
            int(row.get("y1", row.get("Y1", 0))),
            int(row.get("x2", row.get("X2", 0))),
            int(row.get("y2", row.get("Y2", 0))),
        )
        tid = row.get("track_id", row.get("track", row.get("TrackID", -1)))
        try:
            tid = int(tid)
        except (TypeError, ValueError):
            tid = -1
        cls = str(row.get("cls_name", row.get("class", row.get("Class", ""))))
        out.setdefault(frame, []).append({"track_id": tid, "bbox": bbox, "class": cls})
    return out


def _load_events(csv_path: Path) -> dict[int, list[dict[str, Any]]]:
    """Return {frame: [{pet, track_a, track_b, conflict_type}, ...]}."""
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    out: dict[int, list[dict[str, Any]]] = {}
    for _, row in df.iterrows():
        fval = row.get("frame", None)
        if pd.isna(fval):
            continue
        frame = int(fval)
        pet_val = row.get("pet", None)
        try:
            pet_val = float(pet_val) if not pd.isna(pet_val) else None
        except (TypeError, ValueError):
            pet_val = None
        out.setdefault(frame, []).append(
            {
                "pet": pet_val,
                "track_a": row.get("track_a"),
                "track_b": row.get("track_b"),
                "conflict_type": str(row.get("conflict_type", "")),
            }
        )
    return out


def _draw_grid(frame: np.ndarray, cfg_path: str | None) -> np.ndarray:
    if not cfg_path or not Path(cfg_path).exists():
        return frame
    try:
        cfg = json.loads(Path(cfg_path).read_text())
    except (json.JSONDecodeError, OSError):
        return frame
    for cell in cfg.get("cells") or []:
        bbox = cell.get("bbox") if isinstance(cell, dict) else None
        if not bbox or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = (int(v) for v in bbox[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)
    return frame


def _draw_gates(frame: np.ndarray, cfg_path: str | None) -> np.ndarray:
    if not cfg_path or not Path(cfg_path).exists():
        return frame
    try:
        import yaml as _yaml

        cfg = _yaml.safe_load(Path(cfg_path).read_text()) or {}
    except Exception:
        return frame
    for gate in cfg.get("gates") or []:
        start, end = gate.get("start"), gate.get("end")
        if not (start and end and len(start) >= 2 and len(end) >= 2):
            continue
        p1 = (int(start[0]), int(start[1]))
        p2 = (int(end[0]), int(end[1]))
        cv2.line(frame, p1, p2, (0, 255, 255), 2)
        name = gate.get("name", "")
        if name:
            cv2.putText(
                frame,
                str(name),
                (p1[0] + 4, p1[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
    return frame


def _draw_detections(frame: np.ndarray, dets: list[dict[str, Any]]) -> np.ndarray:
    for det in dets:
        x1, y1, x2, y2 = det["bbox"]
        tid = det["track_id"]
        color = _PALETTE[tid % len(_PALETTE)] if tid >= 0 else (200, 200, 200)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"#{tid}" if tid >= 0 else (det.get("class") or "?")
        cv2.putText(
            frame,
            str(label),
            (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return frame


def _draw_event_annotations(frame: np.ndarray, events: list[dict[str, Any]]) -> np.ndarray:
    y = 24
    for evt in events:
        pet = evt.get("pet")
        pet_s = f"PET={pet:.2f}s" if isinstance(pet, (int, float)) else "PET=?"
        label = f"CONFLICT {pet_s} [{evt.get('conflict_type', '')}]"
        cv2.putText(
            frame, label, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA
        )
        y += 26
    return frame


def render_full_video(
    video_path: str,
    events_csv: str,
    out_path: str,
    detections_csv: str | None = None,
    grid_config: str | None = None,
    gate_config: str | None = None,
    max_frames: int | None = None,
    fps: float | None = None,
    log_every: int = 100,
) -> dict[str, Any]:
    """Render an annotated MP4 from a completed pipeline run.

    Returns dict with: out_path, frames_written, events_drawn, detections_drawn.
    """
    vp = Path(video_path)
    op = Path(out_path)
    op.parent.mkdir(parents=True, exist_ok=True)
    if not vp.exists():
        raise FileNotFoundError(f"video not found: {vp}")

    events_by_frame = _load_events(Path(events_csv))
    dets_by_frame = _load_detections(Path(detections_csv)) if detections_csv else {}

    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {vp}")
    src_fps = fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(op), fourcc, float(src_fps), (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            f"cannot open video writer for {op} (mp4v codec); "
            "install ffmpeg or use opencv-python (non-headless)"
        )

    frames_written = 0
    events_drawn = 0
    dets_drawn = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and frames_written >= max_frames:
                break
            frame = _draw_grid(frame, grid_config)
            frame = _draw_gates(frame, gate_config)
            if frames_written in dets_by_frame:
                ds = dets_by_frame[frames_written]
                dets_drawn += len(ds)
                frame = _draw_detections(frame, ds)
            if frames_written in events_by_frame:
                es = events_by_frame[frames_written]
                events_drawn += len(es)
                frame = _draw_event_annotations(frame, es)
            writer.write(frame)
            frames_written += 1
            if log_every and frames_written % log_every == 0:
                log.info(
                    "[viz] frames=%d events=%d dets=%d", frames_written, events_drawn, dets_drawn
                )
    finally:
        cap.release()
        writer.release()

    return {
        "out_path": str(op),
        "frames_written": frames_written,
        "events_drawn": events_drawn,
        "detections_drawn": dets_drawn,
    }
