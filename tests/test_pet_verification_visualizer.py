
import pytest
import numpy as np
import pandas as pd
import json
from pathlib import Path
import cv2

from src.analysis.visualization.pet_verification_visualizer import PETVerificationVisualizer

@pytest.fixture
def sample_event_df():
    data = {
        'event_id': [1],
        'pet': [2.0],
        'frame': [100],
        'track_a': [1],
        'track_b': [2],
        'grid_cell': ['G_A_1'],
        'first_track_id': [1],
        'second_track_id': [2],
        'first_exit_frame': [90],
        'first_exit_time_sec': [3.0],
        'second_entry_frame': [110],
        'second_entry_time_sec': [3.667],
        'site': ['GITI'],
        'world_traj_i': [json.dumps([{'frame': 80, 'x_pixel': 100, 'y_pixel': 200},
                                       {'frame': 90, 'x_pixel': 120, 'y_pixel': 220}])],
        'world_traj_j': [json.dumps([{'frame': 100, 'x_pixel': 150, 'y_pixel': 250},
                                       {'frame': 110, 'x_pixel': 170, 'y_pixel': 270}])],
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_video_path(tmp_path):
    video_path = tmp_path / "dummy.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (320, 240))
    for _ in range(200):
        writer.write(np.zeros((240,320,3), dtype=np.uint8))
    writer.release()
    return str(video_path)

def test_initialization(sample_event_df, sample_video_path, tmp_path):
    csv_path = tmp_path / "events.csv"
    sample_event_df.to_csv(csv_path, index=False)
    viz = PETVerificationVisualizer(str(csv_path), sample_video_path)
    assert viz.df is not None
    assert viz.video_path == Path(sample_video_path)

def test_load_event_success(sample_event_df, sample_video_path, tmp_path):
    csv_path = tmp_path / "events.csv"
    sample_event_df.to_csv(csv_path, index=False)
    viz = PETVerificationVisualizer(str(csv_path), sample_video_path)
    event = viz.load_event(1)
    assert event['event_id'] == 1
    assert event['pet'] == 2.0

def test_load_event_missing(sample_event_df, sample_video_path, tmp_path):
    csv_path = tmp_path / "events.csv"
    sample_event_df.to_csv(csv_path, index=False)
    viz = PETVerificationVisualizer(str(csv_path), sample_video_path)
    with pytest.raises(ValueError):
        viz.load_event(999)

def test_parse_traj_json(sample_event_df, sample_video_path, tmp_path):
    csv_path = tmp_path / "events.csv"
    sample_event_df.to_csv(csv_path, index=False)
    viz = PETVerificationVisualizer(str(csv_path), sample_video_path)
    traj = viz.parse_traj('[{"frame":1, "x_pixel":10, "y_pixel":20}]')
    assert len(traj) == 1

def test_draw_trajectory(sample_event_df, sample_video_path, tmp_path):
    csv_path = tmp_path / "events.csv"
    sample_event_df.to_csv(csv_path, index=False)
    viz = PETVerificationVisualizer(str(csv_path), sample_video_path)
    frame = np.zeros((240,320,3), dtype=np.uint8)
    traj = [{'frame':0,'x_pixel':10,'y_pixel':10},{'frame':1,'x_pixel':20,'y_pixel':20}]
    out = viz.draw_trajectory(frame, traj, (255,0,0), current_frame=1)
    assert out.shape == frame.shape
    assert np.any(out != 0)

def test_draw_grid_cell(sample_event_df, sample_video_path, tmp_path):
    csv_path = tmp_path / "events.csv"
    sample_event_df.to_csv(csv_path, index=False)
    viz = PETVerificationVisualizer(str(csv_path), sample_video_path)
    frame = np.zeros((240,320,3), dtype=np.uint8)
    out = viz.draw_grid_cell(frame, 'G_A_1')
    assert out.shape == frame.shape
    assert np.any(out != 0)

def test_draw_timing_info(sample_event_df, sample_video_path, tmp_path):
    csv_path = tmp_path / "events.csv"
    sample_event_df.to_csv(csv_path, index=False)
    viz = PETVerificationVisualizer(str(csv_path), sample_video_path)
    frame = np.zeros((240,320,3), dtype=np.uint8)
    event = viz.load_event(1)
    out = viz.draw_timing_info(frame, event)
    assert out.shape == frame.shape
    assert np.any(out != 0)

def test_process_frame(sample_event_df, sample_video_path, tmp_path):
    csv_path = tmp_path / "events.csv"
    sample_event_df.to_csv(csv_path, index=False)
    viz = PETVerificationVisualizer(str(csv_path), sample_video_path)
    event = viz.load_event(1)
    frame = np.zeros((240,320,3), dtype=np.uint8)
    out = viz.process_frame(frame, event, current_frame=100)
    assert out.shape == frame.shape
    assert np.any(out != 0)

def test_generate_video(sample_event_df, sample_video_path, tmp_path):
    csv_path = tmp_path / "events.csv"
    sample_event_df.to_csv(csv_path, index=False)
    viz = PETVerificationVisualizer(str(csv_path), sample_video_path)
    output = tmp_path / "out.mp4"
    viz.generate_video(1, str(output), fps=10)
    assert output.exists()
    assert output.stat().st_size > 0
