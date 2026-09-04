
import pandas as pd
import json
import numpy as np
import cv2
from scipy.signal import savgol_filter
from pathlib import Path

class PETVerificationVisualizer:
    """Draws PET conflict event overlays on video frames for reviewer verification."""

    def __init__(self, pet_csv_path, video_path, conflict_zone_radius=40, background_mode='schematic'):
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


    def _schematic_background(self, width=1600, height=720):
        """Return a clean dark canvas for schematic mode."""
        return np.full((height, width, 3), (45, 45, 45), dtype=np.uint8)

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
        pts = self._smooth_points(traj)
        if len(pts) >= 2:
            # Draw segments with anti-aliasing
            for i in range(len(pts)-1):
                cv2.line(frame, pts[i], pts[i+1], color, 2, lineType=cv2.LINE_AA)
        if current_frame is not None:
            # Use original trajectory to find exact current position, then map to nearest smoothed
            valid = [p for p in traj if int(p.get('frame', 0)) <= current_frame]
            if valid:
                last = valid[-1]
                # Find closest smoothed point to original position
                orig_x = int(last.get('x_pixel', last.get('world_x', 0)))
                orig_y = int(last.get('y_pixel', last.get('world_y', 0)))
                min_dist = 1e9; best_pt = pts[-1]
                for p in pts:
                    d = (p[0]-orig_x)**2 + (p[1]-orig_y)**2
                    if d < min_dist:
                        min_dist = d; best_pt = p
                cv2.circle(frame, best_pt, 6, color, -1)
        return frame

    def draw_grid_cell(self, frame, cell_name, center=None, radius=None):
        h, w = frame.shape[:2]
        if center is None:
            center = (w // 2, h // 2)
        if radius is None:
            radius = self.conflict_zone_radius
        # Draw rectangle around the grid cell region
        x1 = max(0, center[0] - radius)
        y1 = max(0, center[1] - radius)
        x2 = min(w, center[0] + radius)
        y2 = min(h, center[1] + radius)
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['grid'], 2, cv2.LINE_AA)
        # Label the cell
        label = f"Cell: {cell_name}"
        cv2.putText(frame, label, (x1 + 5, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['grid'], 2, cv2.LINE_AA)
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
            max_frame = max(frames_a + frames_b)
            # Generate 100 evenly spaced frames across trajectory span
            total_frames = 100
            video_idx_range = list(range(total_frames))
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
            # Cap number of video frames to max_frames by sampling evenly
            if total_video_frames > max_frames:
                video_idx_range = np.linspace(0, total_video_frames - 1, max_frames, dtype=int)
            else:
                video_idx_range = range(total_video_frames)

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

        for video_idx in video_idx_range:
            if self.background_mode == 'schematic':
                frame = self._schematic_background(width, height)
                # Map idx to trajectory frame
                mapped_frame = min_frame + (video_idx / (total_frames - 1)) * (max_frame - min_frame) if total_frames > 1 else min_frame
            else:
                ret, frame = cap.read()
                if not ret:
                    break
                # Sharpen/contrast the real footage background
                frame = self._enhance_background(frame)
                # Map video frame to trajectory frame proportionally
                if total_video_frames > 1:
                    mapped_frame = min_frame + (video_idx / (total_video_frames - 1)) * (max_frame - min_frame)
                else:
                    mapped_frame = min_frame

            # Draw smoothed trajectories and attach current tracker points to smoothed path
            frame = self.draw_trajectory(frame, traj_a, self.colors['track_a'], current_frame=mapped_frame)
            frame = self.draw_trajectory(frame, traj_b, self.colors['track_b'], current_frame=mapped_frame)

            # Use original positions for conflict zone center (or smoothed approximated)
            pos_a = self._get_position_at(traj_a, mapped_frame)
            pos_b = self._get_position_at(traj_b, mapped_frame)
            if pos_a is not None and pos_b is not None:
                center = ((pos_a[0] + pos_b[0]) // 2, (pos_a[1] + pos_b[1]) // 2)
                cv2.circle(frame, center, self.conflict_zone_radius, self.colors['conflict'], 2)

            # Conflict zone between current positions
            if pos_a is not None and pos_b is not None:
                center = ((pos_a[0] + pos_b[0]) // 2, (pos_a[1] + pos_b[1]) // 2)
                # Pulse effect near conflict frame
                if abs(mapped_frame - event.get('frame', 0)) < 5:
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
