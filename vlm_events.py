# analysis/vlm_events.py
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np
import pandas as pd
from diskcache import Cache

from vlm_schema import VLMSafetyAnalysis, VLMBackend
from vlm_backend_stub import EchoStubBackend  # fallback backend
from vlm_hf_backend import HFVisionLanguageBackend


@dataclass
class VLMEventsConfig:
    cache_dir: Path = Path("outputs/vlm_cache")
    collage_grid: Tuple[int, int] = (2, 3)  # 2x3 frames
    frames_per_side: int = 15  # +/- around conflict frame
    fps: float = 30.0  # used only for timestamp overlay


class VLMEventsAnnotator:
    def __init__(self, backend: VLMBackend | None = None, config: VLMEventsConfig | None = None):
        if backend is not None:
            self.backend = backend
        else:
            try:
                self.backend = HFVisionLanguageBackend(
                    model_name="Qwen/Qwen-VL-Chat",
                    device="cuda",
                )
            except Exception:
                # Fallback to stub if HF backend fails to load
                self.backend = EchoStubBackend()
        self.config = config or VLMEventsConfig()
        self.cache = Cache(str(self.config.cache_dir))
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Public API ----------

    def annotate_pets(
        self,
        pet_csv: Path,
        video_path: Path,
        out_csv: Path,
    ) -> None:
        """Annotate PET events with VLM analysis and save extended CSV."""
        df = pd.read_csv(pet_csv)
        rows = []

        for _, row in df.iterrows():
            event_id = int(row["event_id"])
            # Use frame if available; otherwise fallback to center of video window
            base_frame = self._get_base_frame(row)
            frame_start = max(0, base_frame - self.config.frames_per_side)
            frame_end = base_frame + self.config.frames_per_side

            prompt = self._build_prompt(row)
            prompt_hash = self._hash_prompt(prompt)
            cache_key = self._cache_key(
                event_id=event_id,
                frame_range=(frame_start, frame_end),
                prompt_hash=prompt_hash,
            )

            cached = self.cache.get(cache_key)
            if cached is not None:
                vlm_obj = VLMSafetyAnalysis(**cached)
            else:
                collage_path = self._create_collage_for_vlm(
                    video_path=video_path,
                    frames=self._select_frames(frame_start, frame_end),
                    output_path=Path(f"outputs/vlm_collages/event_{event_id}.jpg"),
                )
                raw_text = self.backend.analyze(str(collage_path), prompt)
                vlm_obj = VLMSafetyAnalysis.from_model_response(raw_text)
                self.cache.set(cache_key, vlm_obj.dict(), expire=86400 * 30)

            # Merge VLM fields into row
            for k, v in vlm_obj.dict().items():
                row[f"vlm_{k}"] = v
            rows.append(row)

        out_df = pd.DataFrame(rows)
        out_df.to_csv(out_csv, index=False)

    # ---------- Prompt construction ----------

    def _build_prompt(self, row: pd.Series) -> str:
        """Few-shot prompt with fixed JSON schema."""
        pet = float(row["pet"])
        conflict_type = str(row.get("conflict_type", "unknown"))
        track_a = row.get("track_a", "NA")
        track_b = row.get("track_b", "NA")

        # PET-based severity bins (you can tune these)
        pet_bin_desc = (
            "< 1.5 s: critical, 1.5–3.0 s: moderate, > 3.0 s: lower risk"
        )

        return f"""
You are a traffic safety expert analyzing conflicts at unsignalized intersections.

Task:
You will see a collage image made of multiple frames from a short traffic conflict video.
Use both the visual information and the numeric PET info to produce a JSON object
describing the conflict.

Surrogate Safety Metric:
- PET (Post-Encroachment Time) is a time-based safety surrogate.
- Use this interpretation: {pet_bin_desc}.

Output JSON schema (all fields required):
{{
  "description": str,
  "scenario_type": "rear_end" | "crossing" | "lane_change" | "pedestrian_crossing" | "cyclist_interaction" | "other",
  "violation_type": "none" | "speeding" | "red_light" | "yield_failure" | "lane_discipline" | "pedestrian_right_of_way" | "other",
  "severity": "low" | "medium" | "high" | "critical",
  "contributing_factors": [str, ...],
  "confidence": float between 0 and 1,
  "pet_comparison": "lower" | "similar" | "higher"
}}

Example 1:
Input: PET=0.8s, car from left, cyclist going straight
Output:
{{
  "description": "Car from left fails to yield at crosswalk, forcing cyclist to brake hard.",
  "scenario_type": "crossing",
  "violation_type": "yield_failure",
  "severity": "critical",
  "contributing_factors": ["occluded view", "high approach speed"],
  "confidence": 0.95,
  "pet_comparison": "similar"
}}

Example 2:
Input: PET=2.5s, two cars, one changing lanes
Output:
{{
  "description": "Lane change with adequate gap but slight speed differential.",
  "scenario_type": "lane_change",
  "violation_type": "none",
  "severity": "low",
  "contributing_factors": ["normal traffic flow"],
  "confidence": 0.88,
  "pet_comparison": "higher"
}}

Now analyze this conflict:

Numeric data:
- PET: {pet:.2f} seconds
- Conflict cell: {conflict_type}
- Actors: Track {track_a} and Track {track_b}

Return ONLY valid JSON that matches the schema exactly, with no extra text.
"""

    # ---------- Frame selection & collage ----------

    def _get_base_frame(self, row: pd.Series) -> int:
        """Choose a base frame for the conflict.

        For now, use 'frame' column if present; can later be replaced with
        PET-based min-distance frame logic.
        """
        if "frame" in row and pd.notna(row["frame"]):
            return int(row["frame"])
        # Fallback: arbitrary mid frame (can be improved)
        return 50

    def _select_frames(self, frame_start: int, frame_end: int) -> list[int]:
        """Select frames to sample within [start, end] for collage."""
        total = frame_end - frame_start + 1
        grid_h, grid_w = self.config.collage_grid
        needed = grid_h * grid_w
        if total <= needed:
            return list(range(frame_start, frame_end + 1))

        # Evenly sample 'needed' frames from the interval
        indices = np.linspace(frame_start, frame_end, num=needed, dtype=int)
        return sorted(set(indices.tolist()))

    def _create_collage_for_vlm(
        self,
        video_path: Path,
        frames: Iterable[int],
        output_path: Path,
    ) -> Path:
        """Create a grid collage for VLM with timestamp overlays."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        images = []
        for frame_num in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ok, frame = cap.read()
            if not ok:
                continue
            # Add timestamp overlay
            t_sec = frame_num / self.config.fps
            cv2.putText(
                frame,
                f"t={t_sec:.1f}s",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            images.append(frame)
        cap.release()

        if not images:
            raise RuntimeError(f"No frames captured from {video_path} for collage.")

        h, w = images[0].shape[:2]
        grid_h, grid_w = self.config.collage_grid
        canvas = np.zeros((grid_h * h, grid_w * w, 3), dtype=np.uint8)

        for idx, img in enumerate(images[: grid_h * grid_w]):
            r = idx // grid_w
            c = idx % grid_w
            canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = img

        cv2.imwrite(str(output_path), canvas)
        return output_path

    # ---------- Caching helpers ----------

    def _hash_prompt(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def _cache_key(
        self,
        event_id: int,
        frame_range: Tuple[int, int],
        prompt_hash: str,
    ) -> str:
        return f"{self.backend.model_name}:{event_id}:{frame_range[0]}-{frame_range[1]}:{prompt_hash[:16]}"
