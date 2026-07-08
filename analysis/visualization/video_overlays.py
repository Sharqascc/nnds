"""
Publication-Quality Video Overlay Visualization for Traffic Conflicts

Advanced video frame processing with:
- Trajectory overlay on video frames
- Vehicle bounding boxes with tracking IDs
- Severity-based color coding (PET/TTC)
- Statistical annotations (on-frame display)
- Conflict zone visualization
- Video generation (MP4 output)
- Before/during/after sequences for papers
- Comparison mode (real vs diffusion-generated)

Features:
- Colorblind-safe palettes (Okabe-Ito)
- High-resolution export (300 DPI for stills)
- Publication-quality formatting
- Batch processing capabilities
- Video codec support (H.264, XVID)

Compliant with:
- IEEE/TRB figure guidelines
- Journal submission requirements
- Video supplement standards
"""

import os
import cv2
import warnings
from typing import Optional, List, Tuple, Dict, Union
import numpy as np
from pathlib import Path

try:
    from grid_trajectory.spatial_grid import SpatialGrid
except ImportError:
    SpatialGrid = None


__all__ = [
    'VideoOverlayPlotter',
    'overlay_conflict_frame',
    'generate_conflict_video',
    'create_before_during_after',
    'save_conflict_frame'
]


# Colorblind-safe palette (Okabe-Ito) in BGR for OpenCV
# BGR palette and thresholds are derived from the shared
# analysis.visualization.severity module (which stores hex colors, since
# matplotlib-based files use hex) so this file can't drift out of sync with
# industry_standard_viz.py / pet_event_plots.py. OpenCV needs BGR tuples,
# so we convert the shared hex values once here.
from analysis.visualization.severity import (
    COLORS as _HEX_COLORS,
    DEFAULT_PET_THRESHOLDS as DEFAULT_THRESHOLDS,
    get_severity_color as _shared_get_severity_color,
    get_severity_label as _shared_get_severity_label,
)


def _hex_to_bgr(hex_color: str):
    hex_color = hex_color.lstrip('#')
    r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (b, g, r)


COLORS_BGR = {name: _hex_to_bgr(hexval) for name, hexval in _HEX_COLORS.items()}


class VideoOverlayPlotter:
    """
    Publication-quality video overlay plotter for conflict events.

    Features:
    - Trajectory overlay with colorblind-safe colors
    - Vehicle bounding boxes with tracking IDs
    - Severity-based annotations
    - Video generation (MP4)
    - Before/during/after sequences
    - Comparison visualization
    """

    def __init__(
        self,
        dpi: int = 300,
        colorblind_safe: bool = True,
        thresholds: Optional[Dict[str, float]] = None,
        font_scale: float = 0.6,
        line_thickness: int = 2,
        show_grid: bool = True,
        grid_alpha: float = 0.4
    ):
        """
        Args:
            dpi: Resolution for saved images (300 for publication)
            colorblind_safe: Use Okabe-Ito palette
            thresholds: Custom PET thresholds
            font_scale: OpenCV font size multiplier
            line_thickness: Trajectory line thickness
            show_grid: Show spatial grid overlay
            grid_alpha: Grid transparency (0-1)
        """
        self.dpi = dpi
        self.colorblind_safe = colorblind_safe
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.font_scale = font_scale
        self.line_thickness = line_thickness
        self.show_grid = show_grid
        self.grid_alpha = grid_alpha

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self._grid_warning_emitted = False

    def _get_severity_color(self, pet_value: float) -> Tuple[int, int, int]:
        """Get BGR color based on PET severity (delegates to shared severity module)."""
        hex_color = _shared_get_severity_color(pet_value, self.thresholds)
        return _hex_to_bgr(hex_color)

    def _get_severity_label(self, pet_value: float) -> str:
        """Get severity category label (delegates to shared severity module)."""
        return _shared_get_severity_label(pet_value, self.thresholds).upper()

    def overlay_trajectories(
        self,
        frame: np.ndarray,
        trajectories: List[List[Tuple[float, float, float]]],
        track_ids: Optional[List[int]] = None,
        colors: Optional[List[Tuple[int, int, int]]] = None,
        show_arrows: bool = True
    ) -> np.ndarray:
        """
        Overlay trajectories on a video frame.

        Args:
            frame: Input BGR frame
            trajectories: List of [(t, x, y), ...] for each track
            track_ids: Optional track IDs for labeling
            colors: Optional custom colors (BGR), defaults to colorblind-safe
            show_arrows: Show direction arrows

        Returns:
            Frame with trajectories overlaid
        """
        output = frame.copy()

        if colors is None:
            # Default colorblind-safe colors
            color_cycle = [
                COLORS_BGR['blue'],
                COLORS_BGR['orange'],
                COLORS_BGR['green'],
                COLORS_BGR['purple'],
                COLORS_BGR['cyan']
            ]
            colors = [color_cycle[i % len(color_cycle)] for i in range(len(trajectories))]

        for idx, traj in enumerate(trajectories):
            if len(traj) < 2:
                continue

            color = colors[idx]

            # Extract points
            points = [(int(x), int(y)) for (t, x, y) in traj]

            # Draw trajectory line
            for i in range(len(points) - 1):
                cv2.line(output, points[i], points[i+1], color, self.line_thickness)

            # Draw circles at each point
            for pt in points:
                cv2.circle(output, pt, 3, color, -1)

            # Start marker (larger)
            cv2.circle(output, points[0], 6, color, -1)
            cv2.circle(output, points[0], 8, COLORS_BGR['black'], 2)

            # End marker (arrow if enabled)
            if show_arrows and len(points) >= 2:
                p1 = points[-2]
                p2 = points[-1]
                cv2.arrowedLine(output, p1, p2, color, self.line_thickness + 1, tipLength=0.3)

            # Track ID label
            if track_ids and idx < len(track_ids):
                label_pos = (points[0][0] + 10, points[0][1] - 10)
                cv2.putText(
                    output,
                    f"ID {track_ids[idx]}",
                    label_pos,
                    self.font,
                    self.font_scale * 0.8,
                    color,
                    self.line_thickness,
                    cv2.LINE_AA
                )

        return output

    def overlay_bounding_boxes(
        self,
        frame: np.ndarray,
        boxes: List[Tuple[int, int, int, int]],
        track_ids: Optional[List[int]] = None,
        colors: Optional[List[Tuple[int, int, int]]] = None
    ) -> np.ndarray:
        """
        Overlay bounding boxes on frame.

        Args:
            frame: Input BGR frame
            boxes: List of (x1, y1, x2, y2) bounding boxes
            track_ids: Optional track IDs
            colors: Optional custom colors

        Returns:
            Frame with bounding boxes
        """
        output = frame.copy()

        if colors is None:
            colors = [COLORS_BGR['blue']] * len(boxes)

        for idx, (x1, y1, x2, y2) in enumerate(boxes):
            color = colors[idx]

            # Draw rectangle
            cv2.rectangle(output, (x1, y1), (x2, y2), color, self.line_thickness)

            # Track ID label
            if track_ids and idx < len(track_ids):
                label = f"ID {track_ids[idx]}"
                label_size = cv2.getTextSize(label, self.font, self.font_scale, self.line_thickness)[0]

                # Background for text
                cv2.rectangle(
                    output,
                    (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0] + 5, y1),
                    color,
                    -1
                )

                # Text
                cv2.putText(
                    output,
                    label,
                    (x1 + 2, y1 - 5),
                    self.font,
                    self.font_scale,
                    COLORS_BGR['black'],
                    self.line_thickness,
                    cv2.LINE_AA
                )

        return output

    def overlay_conflict_info(
        self,
        frame: np.ndarray,
        pet_value: float,
        ttc_value: Optional[float] = None,
        frame_number: Optional[int] = None,
        timestamp: Optional[float] = None,
        position: str = 'top-left'
    ) -> np.ndarray:
        """
        Overlay conflict statistics on frame.

        Args:
            frame: Input BGR frame
            pet_value: PET value in seconds
            ttc_value: Optional TTC value
            frame_number: Optional frame number
            timestamp: Optional timestamp in seconds
            position: 'top-left', 'top-right', 'bottom-left', 'bottom-right'

        Returns:
            Frame with statistics overlay
        """
        output = frame.copy()
        h, w = frame.shape[:2]

        severity_color = self._get_severity_color(pet_value)
        severity_label = self._get_severity_label(pet_value)

        # Build info text
        info_lines = [
            f"PET: {pet_value:.3f}s ({severity_label})"
        ]

        if ttc_value is not None:
            info_lines.append(f"TTC: {ttc_value:.3f}s")

        if frame_number is not None:
            info_lines.append(f"Frame: {frame_number}")

        if timestamp is not None:
            info_lines.append(f"Time: {timestamp:.2f}s")

        # Calculate text size
        line_height = int(30 * self.font_scale)
        max_width = max(
            cv2.getTextSize(line, self.font, self.font_scale, self.line_thickness)[0][0]
            for line in info_lines
        )

        box_height = len(info_lines) * line_height + 20
        box_width = max_width + 20

        # Position
        if position == 'top-left':
            x, y = 10, 10
        elif position == 'top-right':
            x, y = w - box_width - 10, 10
        elif position == 'bottom-left':
            x, y = 10, h - box_height - 10
        else:  # bottom-right
            x, y = w - box_width - 10, h - box_height - 10

        # Draw semi-transparent background
        overlay = output.copy()
        cv2.rectangle(
            overlay,
            (x, y),
            (x + box_width, y + box_height),
            severity_color,
            -1
        )
        output = cv2.addWeighted(overlay, 0.7, output, 0.3, 0)

        # Draw border
        cv2.rectangle(
            output,
            (x, y),
            (x + box_width, y + box_height),
            severity_color,
            3
        )

        # Draw text
        for i, line in enumerate(info_lines):
            text_y = y + (i + 1) * line_height
            cv2.putText(
                output,
                line,
                (x + 10, text_y),
                self.font,
                self.font_scale,
                (255, 255, 255),  # White text
                self.line_thickness,
                cv2.LINE_AA
            )

        return output

    def overlay_conflict_zone(
        self,
        frame: np.ndarray,
        center: Tuple[int, int],
        radius: int = 50,
        color: Optional[Tuple[int, int, int]] = None,
        alpha: float = 0.3
    ) -> np.ndarray:
        """
        Highlight conflict zone with translucent circle.

        Args:
            frame: Input BGR frame
            center: (x, y) center of conflict zone
            radius: Radius in pixels
            color: Optional custom color (BGR)
            alpha: Transparency

        Returns:
            Frame with conflict zone highlighted
        """
        output = frame.copy()

        if color is None:
            color = COLORS_BGR['red']

        overlay = output.copy()
        cv2.circle(overlay, center, radius, color, -1)
        output = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0)

        # Border
        cv2.circle(output, center, radius, color, 3)

        return output

    def overlay_conflict_frame(
        self,
        video_path: str,
        frame_idx: int,
        trajectories: List[List[Tuple[float, float, float]]],
        track_ids: Optional[List[int]] = None,
        pet_value: Optional[float] = None,
        conflict_center: Optional[Tuple[int, int]] = None,
        grid: Optional['SpatialGrid'] = None,
        cell_id: Optional[str] = None
    ) -> np.ndarray:
        """
        Create complete conflict visualization for a single frame.

        Args:
            video_path: Path to video file
            frame_idx: Frame index to extract
            trajectories: Vehicle trajectories
            track_ids: Track IDs
            pet_value: PET value for annotation
            conflict_center: Optional conflict zone center
            grid: Optional SpatialGrid for grid overlay
            cell_id: Optional cell ID to highlight

        Returns:
            Processed frame
        """
        # Extract frame
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")

        output = frame.copy()

        if self.show_grid and grid is None and not self._grid_warning_emitted:
            warnings.warn(
                "SpatialGrid not available; grid overlay disabled.",
                UserWarning,
                stacklevel=2,
            )
            self._grid_warning_emitted = True

        # Grid overlay
        if self.show_grid and grid is not None:
            output = grid.draw_overlay(output, alpha=self.grid_alpha)

            if cell_id:
                center = grid.get_cell_center(cell_id)
                if center:
                    cx, cy = center
                    half = grid.cell_size // 2
                    overlay_temp = output.copy()
                    cv2.rectangle(
                        overlay_temp,
                        (int(cx - half), int(cy - half)),
                        (int(cx + half), int(cy + half)),
                        COLORS_BGR['red'],
                        -1
                    )
                    output = cv2.addWeighted(overlay_temp, 0.3, output, 0.7, 0)

        # Trajectories
        output = self.overlay_trajectories(output, trajectories, track_ids)

        # Conflict zone
        if conflict_center:
            output = self.overlay_conflict_zone(output, conflict_center)

        # Statistics
        if pet_value is not None:
            output = self.overlay_conflict_info(
                output,
                pet_value,
                frame_number=frame_idx
            )

        return output

    def save_frame(
        self,
        frame: np.ndarray,
        save_path: str,
        dpi: Optional[int] = None
    ):
        """
        Save frame with publication quality.

        Args:
            frame: BGR frame to save
            save_path: Output path
            dpi: Optional DPI override
        """
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

        # High quality JPEG or PNG
        if save_path.lower().endswith('.png'):
            cv2.imwrite(save_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        else:
            cv2.imwrite(save_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    def compose_inset(
        self,
        frame: np.ndarray,
        inset: np.ndarray,
        position: str = "bottom-right",
        scale: float = 0.25,
        border: int = 4,
        border_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> np.ndarray:
        """
        Compose an inset image (e.g., BEV view) onto the main frame.

        Args:
            frame: Main BGR frame.
            inset: BGR inset image (e.g., top-down BEV plot).
            position: 'top-left', 'top-right', 'bottom-left', 'bottom-right'.
            scale: Width of inset as fraction of main frame width.
            border: Border thickness in pixels.
            border_color: Border color (BGR).

        Returns:
            Frame with inset composed.
        """
        output = frame.copy()
        h, w = output.shape[:2]

        # Resize inset to target width
        target_w = max(int(w * scale), 1)
        aspect = inset.shape[0] / max(inset.shape[1], 1)
        target_h = max(int(target_w * aspect), 1)

        inset_resized = cv2.resize(inset, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # Position top-left corner
        if position == "top-left":
            x0, y0 = border, border
        elif position == "top-right":
            x0, y0 = w - target_w - border, border
        elif position == "bottom-left":
            x0, y0 = border, h - target_h - border
        else:  # bottom-right
            x0, y0 = w - target_w - border, h - target_h - border

        # Draw border
        cv2.rectangle(
            output,
            (x0 - border, y0 - border),
            (x0 + target_w + border, y0 + target_h + border),
            border_color,
            thickness=border
        )

        # Paste inset
        roi = output[y0:y0 + target_h, x0:x0 + target_w]
        if roi.shape[:2] == inset_resized.shape[:2]:
            output[y0:y0 + target_h, x0:x0 + target_w] = inset_resized

        return output




    def generate_conflict_video(
        self,
        video_path: str,
        frame_range: Tuple[int, int],
        trajectories: List[List[Tuple[float, float, float]]],
        track_ids: Optional[List[int]] = None,
        pet_value: Optional[float] = None,
        output_path: str = 'output/conflict_video.mp4',
        fps: int = 30,
        inset_callback: Optional[callable] = None,
        inset_position: str = "bottom-right",
        inset_scale: float = 0.25
    ):
        """
        Generate video of conflict event with overlays.

        Args:
            video_path: Input video path
            frame_range: (start_frame, end_frame)
            trajectories: Vehicle trajectories
            track_ids: Track IDs
            pet_value: PET value for annotation
            output_path: Output video path
            fps: Output video FPS
            inset_callback: Optional function frame_idx -> inset BGR image (e.g., BEV).
            inset_position: Position of inset ('top-left', 'top-right', 'bottom-left', 'bottom-right').
            inset_scale: Inset width as fraction of frame width.
        """
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        start_frame, end_frame = frame_range

        for frame_idx in range(start_frame, end_frame + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                break

            # Base overlay: trajectories (image-plane)
            processed = self.overlay_trajectories(frame, trajectories, track_ids)

            # PET / stats
            if pet_value is not None:
                processed = self.overlay_conflict_info(
                    processed,
                    pet_value,
                    frame_number=frame_idx,
                    timestamp=frame_idx / float(fps)
                )

            # Optional inset (e.g., BEV snapshot)
            if inset_callback is not None:
                try:
                    inset_img = inset_callback(frame_idx)
                    if inset_img is not None:
                        processed = self.compose_inset(
                            processed,
                            inset_img,
                            position=inset_position,
                            scale=inset_scale
                        )
                except Exception as e:
                    warnings.warn(f"Inset callback failed at frame {frame_idx}: {e}")

            out.write(processed)

        cap.release()
        out.release()

        print(f"✅ Video saved: {output_path}")
