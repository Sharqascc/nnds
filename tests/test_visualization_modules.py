
import json
import os
import sys
import warnings
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

# pet_event_plots imports
from src.analysis.visualization.pet_event_plots import (
    EventPlotter,
    get_class_default,
    load_pet_csv,
    compute_timing_from_traj,
    plot_conflict_event,
    plot_multiple_events,
)

# pet_diffusion_plots imports
from src.analysis.visualization.pet_diffusion_plots import (
    _maybe_save,
    plot_pet_like_histogram,
    plot_true_vs_pet_like,
    plot_true_vs_sample_delta,
    plot_bland_altman,
    DiffusionPETPlotter,
)


# ====================== pet_event_plots ======================

def test_load_pet_csv(tmp_path):
    csv = tmp_path / "pet.csv"
    csv.write_text("world_traj_i,world_traj_j,event_id,pet,track_a,track_b,conflict_type\n"
                   '"[[0,1,2],[1,3,4]]","[[0,1,2],[1,3,4]]",1,0.5,1,2,crossing\n')
    df = load_pet_csv(str(csv))
    assert len(df) == 1
    assert isinstance(df.iloc[0]['traj_i'], list)
    assert df.iloc[0]['traj_i'] == [[0,1,2],[1,3,4]]


def test_compute_timing_from_traj():
    df = pd.DataFrame({
        'traj_i': [[[0,0,0],[1,1,1],[2,2,2]]],
        'traj_j': [[[0,0,0],[1,1,1],[2,2,2]]]
    })
    out = compute_timing_from_traj(df)
    assert 'pet_approx' in out.columns
    assert 't_leave_i' in out.columns
    assert 't_enter_j' in out.columns
    assert 'dist_min' in out.columns


def test_get_class_default():
    assert get_class_default(123) == "vehicle"


def test_event_plotter_init_defaults():
    plotter = EventPlotter()
    assert plotter.dpi == 300
    assert plotter.style == "journal"
    assert plotter.colorblind_safe is True
    assert plotter.conflict_zone_radius == 2.0
    assert plotter.arrow_scale == 3.0


def test_event_plotter_setup_style_journal():
    plotter = EventPlotter(style='journal', font_size=12)
    # no exception is enough
    assert plotter.font_size == 12


def test_event_plotter_severity_color():
    from src.analysis.visualization.pet_event_plots import COLORS
    plotter = EventPlotter()
    assert plotter._get_severity_color(0.1) == COLORS["red"]
    assert plotter._get_severity_color(0.7) == COLORS["orange"]
    assert plotter._get_severity_color(1.2) == COLORS["yellow"]
    assert plotter._get_severity_color(2.0) == COLORS["green"]
    assert plotter._get_severity_color(6.0) == COLORS["blue"]


def test_event_plotter_severity_label():
    plotter = EventPlotter()
    assert plotter._get_severity_label(0.1) == "Critical"
    assert plotter._get_severity_label(0.7) == "Serious"
    assert plotter._get_severity_label(1.2) == "Moderate"
    assert plotter._get_severity_label(2.0) == "Slight"
    assert plotter._get_severity_label(6.0) == "Safe"


def test_plot_conflict_event_with_mocks():
    df = pd.DataFrame({
        'event_id': [1],
        'traj_i': [[[0,0,0],[1,1,1],[2,2,2]]],
        'traj_j': [[[0,0,0],[1,1,1],[2,2,2]]],
        'pet': [0.5],
        'track_a': [10],
        'track_b': [20],
        'conflict_type': ['crossing'],
        't_leave_i': [0.1],
        't_enter_j': [0.2],
    })

    plotter = EventPlotter()
    fig = MagicMock()
    ax = MagicMock()
    with patch('matplotlib.pyplot.subplots', return_value=(fig, ax)),          patch('matplotlib.pyplot.tight_layout'),          patch('matplotlib.pyplot.show'):
        result = plotter.plot_conflict_event(df, 1, save_path=None, show_conflict_zone=True, show_velocities=True)
    assert result is fig


def test_plot_conflict_event_save_path(tmp_path):
    df = pd.DataFrame({
        'event_id': [1],
        'traj_i': [[[0,0,0],[1,1,1],[2,2,2]]],
        'traj_j': [[[0,0,0],[1,1,1],[2,2,2]]],
        'pet': [0.5],
        'track_a': [10],
        'track_b': [20],
        'conflict_type': ['crossing'],
        't_leave_i': [0.1],
        't_enter_j': [0.2],
    })

    plotter = EventPlotter()
    fig = MagicMock()
    ax = MagicMock()
    save_path = tmp_path / "event.png"
    with patch('matplotlib.pyplot.subplots', return_value=(fig, ax)),          patch('matplotlib.pyplot.tight_layout'),          patch.object(plotter, '_save_figure', MagicMock()) as save_mock:
        plotter.plot_conflict_event(df, 1, save_path=str(save_path), save_pdf=False)
    save_mock.assert_called_once()


def test_plot_multiple_events_with_mocks(tmp_path):
    df = pd.DataFrame({
        'event_id': [1, 2],
        'traj_i': [[[0,0,0],[1,1,1],[2,2,2]], [[0,0,0],[1,1,1],[2,2,2]]],
        'traj_j': [[[0,0,0],[1,1,1],[2,2,2]], [[0,0,0],[1,1,1],[2,2,2]]],
        'pet': [0.5, 1.2],
        'track_a': [10, 30],
        'track_b': [20, 40],
        'conflict_type': ['crossing', 'head_on'],
        't_leave_i': [0.1, 0.3],
        't_enter_j': [0.2, 0.4],
    })

    plotter = EventPlotter()
    with patch.object(plotter, 'plot_conflict_event') as mock_plot:
        plotter.plot_multiple_events(df, [1,2], save_dir=str(tmp_path/"out"))
    assert mock_plot.call_count == 2


def test_save_figure():
    plotter = EventPlotter()
    fig = MagicMock()
    path = "/tmp/test_fig.png"
    with patch('os.makedirs'):
        plotter._save_figure(fig, path, save_pdf=False)
    fig.savefig.assert_called_once()


# ====================== pet_diffusion_plots ======================

def test_maybe_save_with_path(tmp_path):
    out_path = tmp_path / "subdir" / "plot.png"
    with patch('matplotlib.pyplot.savefig') as savefig_mock:
        _maybe_save(str(out_path), save_pdf=False)
    assert out_path.parent.exists()
    savefig_mock.assert_called_once()


def test_maybe_save_no_path():
    with patch('matplotlib.pyplot.savefig') as savefig_mock:
        _maybe_save(None, save_pdf=False)
    savefig_mock.assert_not_called()


def test_plot_pet_like_histogram_no_data():
    with pytest.warns(UserWarning):
        plot_pet_like_histogram([(None, None)], out_path=None)
    # Should not raise


def test_plot_pet_like_histogram_with_data():
    pairs = [(1.0, 1.5), (2.0, 2.3)]
    with patch('matplotlib.pyplot.subplots') as mock_subplots, \
         patch('matplotlib.pyplot.show'), patch('matplotlib.pyplot.tight_layout'):
        fig = MagicMock()
        ax = MagicMock()
        # hist returns counts, bin_edges, patches (3-tuple)
        ax.hist.return_value = (np.array([1,1]), np.array([0,0.5,1]), [MagicMock(), MagicMock()])
        mock_subplots.return_value = (fig, ax)
        plot_pet_like_histogram(pairs, out_path=None)
    # Should run without error


def test_plot_true_vs_pet_like_no_data():
    with pytest.warns(UserWarning):
        plot_true_vs_pet_like([(1, 1.0, None, None)], out_path=None)
    # Should not raise


def test_plot_true_vs_pet_like_with_data():
    records = [(1, 2.0, 1.5, 1.8), (2, 3.0, 2.5, 2.8)]
    with patch('matplotlib.pyplot.subplots') as mock_subplots, \
         patch('matplotlib.pyplot.show'), patch('matplotlib.pyplot.tight_layout'):
        fig, ax = MagicMock(), MagicMock()
        mock_subplots.return_value = (fig, ax)
        plot_true_vs_pet_like(records, out_path=None)
    # Should run without error


def test_plot_true_vs_sample_delta_no_data():
    with pytest.warns(UserWarning):
        plot_true_vs_sample_delta([(1, 1.0, None, None)], out_path=None)
    # Should not raise


def test_plot_true_vs_sample_delta_with_data():
    records = [(1, 2.0, 1.5, 1.8), (2, 3.0, 2.5, 2.8)]
    with patch('matplotlib.pyplot.subplots') as mock_subplots, \
         patch('matplotlib.pyplot.show'), patch('matplotlib.pyplot.tight_layout'):
        fig, ax = MagicMock(), MagicMock()
        mock_subplots.return_value = (fig, ax)
        plot_true_vs_sample_delta(records, out_path=None)
    # Should run without error


def test_plot_bland_altman_no_data():
    with pytest.warns(UserWarning):
        plot_bland_altman([(1, 1.0, None, None)], out_path=None)
    # Should not raise


def test_plot_bland_altman_with_data():
    records = [(1, 2.0, 1.5, 1.8), (2, 3.0, 2.5, 2.8)]
    with patch('matplotlib.pyplot.subplots') as mock_subplots, \
         patch('matplotlib.pyplot.show'), patch('matplotlib.pyplot.tight_layout'):
        fig, ax = MagicMock(), MagicMock()
        mock_subplots.return_value = (fig, ax)
        plot_bland_altman(records, out_path=None)
    # Should run without error


def test_diffusion_plotter_init_default():
    plotter = DiffusionPETPlotter()
    assert plotter.style == 'default'
    assert plotter.dpi == 300
    assert plotter.save_pdf is True


def test_diffusion_plotter_init_custom_style():
    with patch('matplotlib.pyplot.style.use') as style_mock:
        DiffusionPETPlotter(style='seaborn', dpi=150, save_pdf=False)
    style_mock.assert_called_once_with('seaborn')


def test_diffusion_plotter_plot_all(tmp_path):
    plotter = DiffusionPETPlotter()
    with patch('src.analysis.visualization.pet_diffusion_plots.plot_pet_like_histogram') as p1, \
         patch('src.analysis.visualization.pet_diffusion_plots.plot_true_vs_pet_like') as p2, \
         patch('src.analysis.visualization.pet_diffusion_plots.plot_true_vs_sample_delta') as p3, \
         patch('src.analysis.visualization.pet_diffusion_plots.plot_residual_analysis') as p4, \
         patch('src.analysis.visualization.pet_diffusion_plots.plot_bland_altman') as p5:
        plotter.plot_all(pet_pairs=[(1,2)], records=[(1,1,1,1)], out_dir=str(tmp_path/"out"))
    assert p1.called and p2.called and p3.called and p4.called and p5.called
