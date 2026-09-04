
import pandas as pd
import json
import numpy as np
import cv2
from scipy.signal import savgol_filter
from pathlib import Path

class PETVerificationVisualizer:
    """Draws PET conflict event overlays on video frames for reviewer verification."""

    def __init__(self, pet_csv_path, video_path, conflict_zone_radius=40, background_mode='schematic', spatial_grid=None):
        self.df = pd.read_csv(pet_csv_path)
        self.video_path = Path(video_path)
        self.conflict_zone_radius = conflict_zone_radius
        self.colors = {
            'track_a': (255, 0, 0),
            'track_b': (0, 165, 255),
            'grid': (0, 255, 255),
            'conflict': (0, 0, 255),
            'text': (255, 255, 255),
        }
        self.background_mode = background_mode
        self.spatial_grid = spatial_grid


    def _schematic_background(self, width=1600, height=720):
        """Return a clean dark canvas for schematic mode."""
        return np.full((height, width, 3), (240, 240, 240), dtype=np.uint8)

    def load_event(self, event_id):
        rows = self.df[self.df['event_id'] == event_id]
        if rows.empty:
            raise ValueError(f"No event with event_id={event_id}")
        return rows.iloc[0]

    def parse_traj(self, json_str):
        if isinstance(json_str, list):
            return json_str
        if isinstance(json_str, str):
            try:
                return json.loads(json_str)
            except Exception:
                pass
            try:
                import ast
                return ast.literal_eval(json_str)
            except Exception:
                return []
        return []

    def _smooth_points(self, traj, window=9, polyorder=3):
        """Return smoothed trajectory points using Savitzky-Golay filter."""
        if len(traj) < 3:
            return [(int(p.get('x_pixel', p.get('world_x', 0))),
                     int(p.get('y_pixel', p.get('world_y', 0)))) for p in traj]
        xs = [p.get('x_pixel', p.get('world_x', 0)) for p in traj]
        ys = [p.get('y_pixel', p.get('world_y', 0)) for p in traj]
        n = len(traj)
        # Ensure window_length is odd and <= n
        w = min(window, n if n % 2 == 1 else n - 1)
        if w < polyorder + 2:
            # Fallback to moving average
            kernel = np.ones(w) / w
            xs_s = np.convolve(xs, kernel, mode='same')
            ys_s = np.convolve(ys, kernel, mode='same')
        else:
            xs_s = savgol_filter(xs, window_length=w, polyorder=polyorder, mode='interp')
            ys_s = savgol_filter(ys, window_length=w, polyorder=polyorder, mode='interp')
        # Keep endpoints original
        xs_s[0], xs_s[-1] = xs[0], xs[-1]
        ys_s[0], ys_s[-1] = ys[0], ys[-1]
        return [(int(x), int(y)) for x, y in zip(xs_s, ys_s)]

    def draw_trajectory(self, frame, traj, color, current_frame=None):
        if current_frame is not None:
            # Only keep points up to current frame for animation
            traj = [p for p in traj if int(p.get('frame', 0)) <= current_frame]
        pts = self._smooth_points(traj)
        if len(pts) >= 2:
            for i in range(len(pts)-1):
                cv2.line(frame, pts[i], pts[i+1], color, 3, lineType=cv2.LINE_AA)
        if pts:
            # Draw current position at last point
            last = pts[-1]
            h, w = frame.shape[:2]
            x = max(0, min(w-1, last[0]))
            y = max(0, min(h-1, last[1]))
            cv2.circle(frame, (x, y), 6, color, -1)
        return frame

    def draw_grid_cell(self, frame, cell_name, center=None, radius=None):
        h, w = frame.shape[:2]
        if center is None:
            center = (w // 2, h // 2)
        if radius is None:
            radius = self.conflict_zone_radius
        x1 = max(0, center[0] - radius)
        y1 = max(0, center[1] - radius)
        x2 = min(w, center[0] + radius)
        y2 = min(h, center[1] + radius)

        # 1) Translucent red fill for high visibility
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
        frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

        # 2) Thick red border
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4, cv2.LINE_AA)

        # 3) Label with dark background for contrast
        label = f"Cell: {cell_name}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        label_y = max(20, y1 - 10)
        # Black rectangle behind text
        cv2.rectangle(frame, (x1, label_y - th - baseline), (x1 + tw + 10, label_y + baseline),
                      (0, 0, 0), -1)
        cv2.putText(frame, label, (x1 + 5, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        return frame


    def _draw_text_background(self, frame, top_left, bottom_right, alpha=0.5):
        """Draw a semi-transparent black rectangle on a copy of the ROI."""
        x1, y1 = top_left
        x2, y2 = bottom_right
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return frame
        roi = frame[y1:y2, x1:x2]
        overlay = np.zeros_like(roi)
        overlay[:] = (0, 0, 0)  # black
        blended = cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0)
        frame[y1:y2, x1:x2] = blended
        return frame

    def draw_timing_info(self, frame, event):
        info = [
            f"Event {event['event_id']}  PET={event['pet']:.3f}s",
            f"First: track {event['first_track_id']} exit frame {event['first_exit_frame']}",
            f"Second: track {event['second_track_id']} entry frame {event['second_entry_frame']}",
            f"Time A exit: {event['first_exit_time_sec']:.2f}s",
            f"Time B entry: {event['second_entry_time_sec']:.2f}s",
            f"Site: {event['site']}  Cell: {event['grid_cell']}",
        ]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        line_height = 20
        # Calculate text block size
        (max_text_w, _), _ = cv2.getTextSize(max(info, key=len), font, font_scale, thickness)
        box_h = line_height * len(info) + 10
        box_w = max_text_w + 20
        box_x, box_y = 10, 10
        frame = self._draw_text_background(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), alpha=0.6)
        y = box_y + line_height
        for line in info:
            cv2.putText(frame, line, (box_x + 10, y), font, font_scale, self.colors['text'], thickness, cv2.LINE_AA)
            y += line_height
        return frame

    def process_frame(self, frame, event, current_frame):
        traj_i = self.parse_traj(event['world_traj_i'])
        traj_j = self.parse_traj(event['world_traj_j'])
        frame = self.draw_trajectory(frame, traj_i, self.colors['track_a'], current_frame)
        frame = self.draw_trajectory(frame, traj_j, self.colors['track_b'], current_frame)
        frame = self.draw_grid_cell(frame, event['grid_cell'])
        frame = self.draw_timing_info(frame, event)
        return frame


    def _enhance_background(self, frame):
        """Apply unsharp masking and CLAHE to make source video less blurry."""
        # Unsharp mask
        blurred = cv2.GaussianBlur(frame, (0, 0), 3.0)
        frame = cv2.addWeighted(frame, 1.5, blurred, -0.5, 0)
        # Contrast enhancement via CLAHE on L channel in LAB
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        lab2 = cv2.merge((l2, a, b))
        return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    def generate_video(self, event_id, output_path, fps=10, max_frames=200):
        event = self.load_event(event_id)
        traj_a = self.parse_traj(event.get('traj_a_json', event.get('world_traj_i')))
        traj_b = self.parse_traj(event.get('traj_b_json', event.get('world_traj_j')))
        if not traj_a and not traj_b:
            raise ValueError(f"No trajectory data for event {event_id}")

        # Schematic mode: no blurry video, use dark canvas
        if self.background_mode == 'schematic':
            width, height = 1600, 720
            frames_a = [int(p.get('frame', 0)) for p in traj_a]
            frames_b = [int(p.get('frame', 0)) for p in traj_b]
            min_frame = min(frames_a + frames_b)
            conflict_frame = int(event.get('frame', max(frames_a + frames_b)))
            # Use conflict frame as the end (or max if conflict not set)
            end_frame = conflict_frame if conflict_frame > min_frame else max(frames_a + frames_b)
            total_span = end_frame - min_frame + 1
            # Determine step to keep <= max_frames
            if total_span > max_frames:
                step = total_span / (max_frames - 1)
                video_idx_range = [min_frame + int(i * step) for i in range(max_frames)]
            else:
                video_idx_range = list(range(min_frame, end_frame + 1))
            total_frames = len(video_idx_range)
        else:
            # Open real video (optional; blurry source not used by default)
            cap = cv2.VideoCapture(str(self.video_path))
            if not cap.isOpened():
                raise RuntimeError("Could not open source video")
            total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_video_frames == 0:
                cap.release()
                raise RuntimeError("Video has no frames")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # Center the sampling window around the conflict frame to avoid missing it.
            conflict_frame = int(event.get('frame', 0))
            half_window = max_frames // 2
            start = max(0, conflict_frame - half_window)
            end = min(total_video_frames, conflict_frame + half_window)
            available = end - start
            if available > max_frames:
                # Sample within the conflict-centered window
                video_idx_range = np.linspace(start, end - 1, max_frames, dtype=int)
            else:
                video_idx_range = list(range(start, end))

        # Determine trajectory frame range
        frames_a = [int(p.get('frame', 0)) for p in traj_a]
        frames_b = [int(p.get('frame', 0)) for p in traj_b]
        all_frames = frames_a + frames_b
        if not all_frames:
            cap.release()  # pragma: no cover
            raise ValueError("Trajectory frames are empty")  # pragma: no cover
        min_frame, max_frame = min(all_frames), max(all_frames)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not out.isOpened():
            if hasattr(self, 'background_mode') and self.background_mode != 'schematic':
                cap.release()
            raise RuntimeError("Could not open VideoWriter")

        for frame_idx in video_idx_range:
            if self.background_mode == 'schematic':
                frame = self._schematic_background(width, height)
            else:
                ret, frame = cap.read()
                if not ret:
                    break
                # Sharpen/contrast the real footage background
                frame = self._enhance_background(frame)
                # frame_idx is already the current loop variable from video_idx_range

            # Draw full spatial grid if available
            if self.spatial_grid is not None:
                frame = self.spatial_grid.draw_overlay(frame, alpha=0.3)

            # Draw smoothed trajectories and attach current tracker points to smoothed path
            frame = self.draw_trajectory(frame, traj_a, self.colors['track_a'], current_frame=frame_idx)
            frame = self.draw_trajectory(frame, traj_b, self.colors['track_b'], current_frame=frame_idx)

            # Use original positions for conflict zone center (or smoothed approximated)
            pos_a = self._get_position_at(traj_a, frame_idx)
            pos_b = self._get_position_at(traj_b, frame_idx)
            if pos_a is not None and pos_b is not None:
                center = ((pos_a[0] + pos_b[0]) // 2, (pos_a[1] + pos_b[1]) // 2)
                cv2.circle(frame, center, self.conflict_zone_radius, self.colors['conflict'], 2)

            # Conflict zone between current positions
            if pos_a is not None and pos_b is not None:
                center = ((pos_a[0] + pos_b[0]) // 2, (pos_a[1] + pos_b[1]) // 2)
                # Pulse effect near conflict frame
                if abs(frame_idx - event.get('frame', 0)) < 5:
                    thickness = 5
                    alpha = 0.5
                else:
                    thickness = 3
                    alpha = 0.8
                # Outer glow
                cv2.circle(frame, center, self.conflict_zone_radius + 8, (0, 0, 255), 2, cv2.LINE_AA)
                # Main circle
                cv2.circle(frame, center, self.conflict_zone_radius, self.colors['conflict'], thickness, cv2.LINE_AA)
                # Filled translucent inner circle
                overlay = frame.copy()
                cv2.circle(overlay, center, self.conflict_zone_radius, self.colors['conflict'], -1)
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                # Draw grid cell rectangle and label
                frame = self.draw_grid_cell(frame, event.get('grid_cell', ''), center=center, radius=self.conflict_zone_radius)

            # Timing and event info
            frame = self.draw_timing_info(frame, event)
            out.write(frame)

        if self.background_mode != 'schematic':
            cap.release()
        out.release()
        return str(output_path)

    def _get_position_at(self, traj, frame_idx):
        """Return (x_pixel, y_pixel) at or before frame_idx, or None."""
        valid = [p for p in traj if int(p.get('frame', 0)) <= frame_idx]
        if not valid:
            return None
        last = valid[-1]
        return (int(last.get('x_pixel', last.get('world_x', 0))),
                int(last.get('y_pixel', last.get('world_y', 0))))
