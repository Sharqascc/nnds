import sys
from pathlib import Path
import pandas as pd
import numpy as np

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))

from src.analysis.grid_trajectory.uvh_coco_fused_grid_pet import _split_tracks_by_gaps, TrackPoint
from scripts.validate_outputs import validate_detections, validate_pet, DETECTION_COLUMNS, PET_COLUMNS

class TestTrackSplitter:
    def test_splits_on_large_jump(self):
        pts = [
            TrackPoint(frame=0, x=0, y=0, cls_id=0, cls_name='car', conf=0.9),
            TrackPoint(frame=1, x=1, y=1, cls_id=0, cls_name='car', conf=0.9),
            TrackPoint(frame=50, x=100, y=100, cls_id=0, cls_name='car', conf=0.9),  # jump
            TrackPoint(frame=51, x=101, y=101, cls_id=0, cls_name='car', conf=0.9),
        ]
        tracks = {1: pts}
        split = _split_tracks_by_gaps(tracks, max_frame_gap=5, max_spatial_jump=30.0, prediction_tolerance=0.0)
        # Expect at least two segments
        assert len(split) >= 2, f"Expected split tracks, got {len(split)}"

    def test_no_split_when_continuous(self):
        pts = [
            TrackPoint(frame=0, x=0, y=0, cls_id=0, cls_name='car', conf=0.9),
            TrackPoint(frame=1, x=2, y=2, cls_id=0, cls_name='car', conf=0.9),
            TrackPoint(frame=2, x=4, y=4, cls_id=0, cls_name='car', conf=0.9),
        ]
        tracks = {2: pts}
        split = _split_tracks_by_gaps(tracks, max_frame_gap=5, max_spatial_jump=30.0, prediction_tolerance=0.0)
        assert len(split) == 1, f"Expected no split, got {len(split)}"

class TestValidationHelpers:
    def test_validate_detections_columns(self):
        df = pd.DataFrame({
            'frame': [0], 'track_id': [1], 'class_id': [0], 'class_name': ['car'],
            'conf': [0.9], 'x1': [10], 'y1': [20], 'x2': [50], 'y2': [70],
            'cx': [30], 'cy': [45], 'source': ['uvh26']
        })
        problems = validate_detections(df)
        assert problems == [], f"Expected no problems, got {problems}"

    def test_validate_pet_positive_pet(self):
        pet_df = pd.DataFrame({
            'event_id': [1], 'pet': [1.5], 'frame': [10], 'track_a': [1], 'track_b': [2],
            'conflict_type': ['image_intersection'], 'grid_cell': ['CELL_A_1'],
            'track_a_entry_frame': [5], 'track_a_exit_frame': [10],
            'track_b_entry_frame': [7], 'track_b_exit_frame': [12],
            'world_traj_i': ['track_1'], 'world_traj_j': ['track_2'],
            'traj_a_json': ['[{"frame":5,"x_pixel":10,"y_pixel":20,"world_x":100,"world_y":200},{"frame":6,"x_pixel":11,"y_pixel":21,"world_x":101,"world_y":201}]'],
            'traj_b_json': ['[{"frame":7,"x_pixel":15,"y_pixel":25,"world_x":110,"world_y":210},{"frame":8,"x_pixel":16,"y_pixel":26,"world_x":111,"world_y":211}]'],
        })
        det_df = pd.DataFrame({'track_id': [1, 2]})
        problems = validate_pet(pet_df, det_df, video_frames=100)
        assert problems == [], f"Expected no problems, got {problems}"

    def test_validate_pet_negative_pet_caught(self):
        pet_df = pd.DataFrame({
            'event_id': [1], 'pet': [-0.5], 'frame': [10], 'track_a': [1], 'track_b': [2],
            'conflict_type': ['image_intersection'], 'grid_cell': ['CELL_A_1'],
            'track_a_entry_frame': [5], 'track_a_exit_frame': [10],
            'track_b_entry_frame': [7], 'track_b_exit_frame': [12],
            'world_traj_i': ['track_1'], 'world_traj_j': ['track_2'],
            'traj_a_json': ['[]'], 'traj_b_json': ['[]'],
        })
        det_df = pd.DataFrame({'track_id': [1, 2]})
        problems = validate_pet(pet_df, det_df, video_frames=100)
        assert any('PET <= 0' in p for p in problems), f"Expected PET<=0 problem, got {problems}"
