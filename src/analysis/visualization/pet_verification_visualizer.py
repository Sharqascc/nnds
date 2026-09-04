
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

    def draw_trajectory(self, frame, traj, color, current_frame=None):
        pts = [(int(p.get('x_pixel', p.get('world_x', 0))),
                int(p.get('y_pixel', p.get('world_y', 0)))) for p in traj]
        if len(pts) >= 2:
            cv2.polylines(frame, [np.array(pts)], False, color, 2)
        if current_frame is not None:
            valid = [p for p in traj if int(p.get('frame', 0)) <= current_frame]
            if valid:
                last = valid[-1]
                pos = (int(last.get('x_pixel', last.get('world_x', 0))),
                       int(last.get('y_pixel', last.get('world_y', 0))))
                cv2.circle(frame, pos, 6, color, -1)
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
        cap = cv2.VideoCapture(str(self.video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            cap.release()
            raise RuntimeError("Video has no frames")
        conflict_frame = int(event['frame'])
        start = max(0, conflict_frame - 25)
        end = min(total - 1, conflict_frame + 25)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        if not out.isOpened():
            cap.release()
            raise RuntimeError("Could not open VideoWriter")
        for idx in range(start, end + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame = self.process_frame(frame, event, idx)
            out.write(frame)
        cap.release()
        out.release()
        return output_path
