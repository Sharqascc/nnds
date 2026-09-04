
import pandas as pd
import json
import numpy as np
import cv2
from pathlib import Path

class PETVerificationVisualizer:
    """Draws PET conflict event overlays on video frames for reviewer verification."""

    def __init__(self, pet_csv_path, video_path, conflict_zone_radius=40):
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

    def _smooth_points(self, traj, sigma=2.0):
        """Return smoothed trajectory points (x,y) using Gaussian kernel."""
        if len(traj) < 3:
            return [(int(p.get('x_pixel', p.get('world_x', 0))),
                     int(p.get('y_pixel', p.get('world_y', 0)))) for p in traj]
        xs = [p.get('x_pixel', p.get('world_x', 0)) for p in traj]
        ys = [p.get('y_pixel', p.get('world_y', 0)) for p in traj]
        # Choose kernel size <= len(traj) and odd
        n = len(traj)
        kernel_size = max(3, int(6 * sigma))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = min(kernel_size, n if n % 2 == 1 else n - 1)
        if kernel_size < 3:
            kernel_size = 3
        kernel = np.exp(-0.5 * (np.arange(kernel_size) - kernel_size//2)**2 / sigma**2)
        kernel /= kernel.sum()
        xs_smooth = np.convolve(xs, kernel, mode='same')
        ys_smooth = np.convolve(ys, kernel, mode='same')
        # Keep endpoints original to avoid edge artifacts
        xs_smooth[0] = xs[0]; xs_smooth[-1] = xs[-1]
        ys_smooth[0] = ys[0]; ys_smooth[-1] = ys[-1]
        return [(int(x), int(y)) for x,y in zip(xs_smooth, ys_smooth)]

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

    def draw_grid_cell(self, frame, cell_name):
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        cv2.circle(frame, center, self.conflict_zone_radius, self.colors['conflict'], 2)
        cv2.putText(frame, f"Cell: {cell_name}", (center[0] - 60, center[1] - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['grid'], 2)
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
        y = 20
        for line in info:
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
            y += 20
        return frame

    def process_frame(self, frame, event, current_frame):
        traj_i = self.parse_traj(event['world_traj_i'])
        traj_j = self.parse_traj(event['world_traj_j'])
        frame = self.draw_trajectory(frame, traj_i, self.colors['track_a'], current_frame)
        frame = self.draw_trajectory(frame, traj_j, self.colors['track_b'], current_frame)
        frame = self.draw_grid_cell(frame, event['grid_cell'])
        frame = self.draw_timing_info(frame, event)
        return frame

    def generate_video(self, event_id, output_path, fps=10):
        event = self.load_event(event_id)
        traj_a = self.parse_traj(event.get('traj_a_json', event.get('world_traj_i')))
        traj_b = self.parse_traj(event.get('traj_b_json', event.get('world_traj_j')))
        if not traj_a and not traj_b:
            raise ValueError(f"No trajectory data for event {event_id}")

        # Open real video
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError("Could not open source video")
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_video_frames == 0:
            cap.release()
            raise RuntimeError("Video has no frames")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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
            cap.release()
            raise RuntimeError("Could not open VideoWriter")

        for video_idx in range(total_video_frames):
            ret, frame = cap.read()
            if not ret:
                break

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
                cv2.circle(frame, center, self.conflict_zone_radius, self.colors['conflict'], 2)

            # Timing and event info
            frame = self.draw_timing_info(frame, event)
            out.write(frame)

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
