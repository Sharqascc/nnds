from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.analysis.visualization.industry_standard_viz import (
    SSMPlotter,
    plot_comparative_boxplot,
    plot_conflict_density_map,
    plot_correlation_heatmap,
    plot_cumulative_distribution,
    plot_pet_distribution,
    plot_severity_scatter,
    plot_temporal_heatmap,
    plot_ttc_time_series,
)


@pytest.fixture
def plotter():
    return SSMPlotter(style="journal", dpi=100)


def make_axes():
    ax = MagicMock()
    # Common return values
    ax.hist.return_value = (np.array([1, 2]), np.array([0, 0.5, 1]), [MagicMock(), MagicMock()])
    return ax


def patch_plotting():
    """Create a context manager with common matplotlib patches."""

    @contextmanager
    def _patch_ctx():
        with (
            patch("matplotlib.pyplot.subplots", return_value=(MagicMock(), make_axes())),
            patch("matplotlib.pyplot.tight_layout", MagicMock()),
            patch("matplotlib.pyplot.show", MagicMock()),
            patch("matplotlib.pyplot.colorbar", MagicMock(return_value=MagicMock())),
            patch("matplotlib.pyplot.savefig", MagicMock()),
        ):
            yield

    return _patch_ctx()


# ---------------- validate_ssm_data ----------------


def test_validate_ssm_data_none(plotter):
    res = plotter.validate_ssm_data(None, "PET")
    assert res["valid"] is False
    assert any("None" in e for e in res["errors"])


def test_validate_ssm_data_2d(plotter):
    res = plotter.validate_ssm_data(np.array([[1, 2], [3, 4]]), "PET")
    assert res["valid"] is False
    assert any("1D" in e for e in res["errors"])


def test_validate_ssm_data_empty(plotter):
    res = plotter.validate_ssm_data(np.array([]), "PET")
    assert res["valid"] is False


def test_validate_ssm_data_nan_inf(plotter):
    data = np.array([1.0, np.nan, np.inf, 2.0])
    res = plotter.validate_ssm_data(data, "PET")
    assert res["valid"] is True
    assert len(res["warnings"]) >= 2
    assert len(res["clean_data"]) == 2


def test_validate_ssm_data_negative_not_allowed(plotter):
    data = np.array([1.0, -2.0, 3.0])
    res = plotter.validate_ssm_data(data, "PET", allow_negative=False)
    assert res["valid"] is True
    assert any("negative" in w for w in res["warnings"])


def test_validate_ssm_data_all_invalid(plotter):
    data = np.array([np.nan, np.inf])
    res = plotter.validate_ssm_data(data, "PET")
    assert res["valid"] is False
    assert any("No valid" in e for e in res["errors"])


def test_validate_ssm_data_positive(plotter):
    data = np.array([1.0, 2.0, 3.0])
    res = plotter.validate_ssm_data(data, "PET", allow_negative=True)
    assert res["valid"] is True
    assert len(res["clean_data"]) == 3
    assert res["removal_rate"] == 0.0


# ---------------- _format_p_value ----------------


def test_format_p_value_ranges():
    p = SSMPlotter()
    assert p._format_p_value(0.0001) == "p < 0.001***"
    assert p._format_p_value(0.005) == "p = 0.005**"
    assert p._format_p_value(0.02) == "p = 0.020*"
    assert p._format_p_value(0.1) == "p = 0.100 (ns)"


# ---------------- _save_figure ----------------


def test_save_figure(plotter):
    fig = MagicMock()
    path = "/tmp/fig.png"
    plotter._save_figure(fig, path)
    # Should call savefig twice (PNG and PDF)
    assert fig.savefig.call_count == 2


# ---------------- plot_pet_distribution ----------------


def test_plot_pet_distribution_invalid(plotter):
    with pytest.raises(ValueError):
        plotter.plot_pet_distribution(np.array([np.nan, np.inf]))


def test_plot_pet_distribution_valid(plotter):
    data = np.array([0.1, 0.6, 1.2, 2.0, 6.0])
    with patch_plotting():
        fig = plotter.plot_pet_distribution(data, show_kde=True, save_path=None)
    assert fig is not None


def test_plot_pet_distribution_save_path(plotter, tmp_path):
    data = np.array([0.1, 0.6, 1.2, 2.0, 6.0])
    save_path = tmp_path / "pet.png"
    with patch_plotting():
        fig = plotter.plot_pet_distribution(data, show_kde=False, save_path=str(save_path))
    assert fig is not None


# ---------------- plot_ttc_time_series ----------------


def test_plot_ttc_time_series_invalid(plotter):
    with pytest.raises(ValueError):
        plotter.plot_ttc_time_series(np.array([0, 1]), np.array([np.nan]))


def test_plot_ttc_time_series_valid(plotter):
    ttc = np.array([2.0, 1.0, 4.0, 0.2, 5.0])
    timestamps = np.arange(len(ttc))
    with patch_plotting():
        fig = plotter.plot_ttc_time_series(timestamps, ttc, save_path=None)
    assert fig is not None


def test_plot_ttc_time_series_mismatched_timestamps(plotter):
    ttc = np.array([2.0, 1.0, 4.0])
    timestamps = np.array([10, 20])  # wrong length
    with patch_plotting():
        fig = plotter.plot_ttc_time_series(timestamps, ttc)
    assert fig is not None


# ---------------- plot_severity_scatter ----------------


def test_plot_severity_scatter(plotter):
    pet = np.array([0.5, 1.5, 2.5, 3.5])
    ttc = np.array([1.0, 2.0, 3.0, 4.0])
    with patch_plotting():
        fig = plotter.plot_severity_scatter(pet, ttc, add_regression=True, save_path=None)
    assert fig is not None


# ---------------- plot_comparative_boxplot ----------------


def test_plot_comparative_boxplot_less_than_2_groups(plotter):
    with pytest.raises(ValueError):
        plotter.plot_comparative_boxplot({"A": np.array([1, 2, 3])})


def test_plot_comparative_boxplot(plotter):
    groups = {"A": np.array([1, 2, 3]), "B": np.array([2, 3, 4])}
    with patch_plotting(), patch("scipy.stats.ttest_ind", return_value=(1.0, 0.01)):
        fig = plotter.plot_comparative_boxplot(groups, show_stats=True, save_path=None)
    assert fig is not None


# ---------------- plot_conflict_density_map ----------------


def test_plot_conflict_density_map_default(plotter):
    pet = np.array([0.1, 0.6, 1.2, 2.0, 6.0])
    with patch_plotting():
        fig = plotter.plot_conflict_density_map(pet)
    assert fig is not None


def test_plot_conflict_density_map_custom_bands(plotter):
    pet = np.array([0.1, 0.6, 1.2, 2.0, 6.0])
    custom_bands = [0, 1, 3, 10]
    with patch_plotting():
        fig = plotter.plot_conflict_density_map(pet, custom_bands=custom_bands)
    assert fig is not None


# ---------------- plot_cumulative_distribution ----------------


def test_plot_cumulative_distribution_pet(plotter):
    groups = {"A": np.array([1, 2, 3]), "B": np.array([2, 3, 4])}
    with patch_plotting():
        fig = plotter.plot_cumulative_distribution(groups, metric_name="PET")
    assert fig is not None


def test_plot_cumulative_distribution_ttc(plotter):
    groups = {"A": np.array([1, 2, 3])}
    with patch_plotting():
        fig = plotter.plot_cumulative_distribution(groups, metric_name="TTC")
    assert fig is not None


# ---------------- Standalone convenience functions ----------------


def test_standalone_pet_distribution():
    with patch_plotting():
        fig = plot_pet_distribution(np.array([1, 2, 3]))
    assert fig is not None


def test_standalone_ttc_time_series():
    with patch_plotting():
        fig = plot_ttc_time_series(np.arange(3), np.array([1, 2, 3]))
    assert fig is not None


def test_standalone_severity_scatter():
    with patch_plotting():
        fig = plot_severity_scatter(np.array([1, 2, 3]), np.array([1, 2, 3]))
    assert fig is not None


def test_standalone_conflict_density():
    with patch_plotting():
        fig = plot_conflict_density_map(np.array([1, 2, 3]))
    assert fig is not None


def test_standalone_comparative_boxplot():
    with patch_plotting(), patch("scipy.stats.ttest_ind", return_value=(1.0, 0.01)):
        fig = plot_comparative_boxplot({"A": np.array([1, 2, 3]), "B": np.array([2, 3, 4])})
    assert fig is not None


def test_standalone_cumulative_distribution():
    with patch_plotting():
        fig = plot_cumulative_distribution({"A": np.array([1, 2, 3])})
    assert fig is not None


# ---------------- plot_correlation_heatmap ----------------


def test_plot_correlation_heatmap_with_pandas(monkeypatch):
    # Ensure HAS_PANDAS is True
    monkeypatch.setattr("src.analysis.visualization.industry_standard_viz.HAS_PANDAS", True)
    data = {"A": np.array([1, 2, 3]), "B": np.array([2, 3, 4])}
    with patch_plotting(), patch("matplotlib.pyplot.colorbar", return_value=MagicMock()):
        fig = plot_correlation_heatmap(data)
    assert fig is not None


def test_plot_correlation_heatmap_no_pandas(monkeypatch):
    monkeypatch.setattr("src.analysis.visualization.industry_standard_viz.HAS_PANDAS", False)
    data = {"A": np.array([1, 2, 3]), "B": np.array([2, 3, 4])}
    with pytest.raises(ImportError):
        plot_correlation_heatmap(data)


# ---------------- plot_temporal_heatmap ----------------


def test_plot_temporal_heatmap():
    timestamps = np.array([0, 1, 2, 3, 4])
    pet = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    with patch_plotting():
        fig = plot_temporal_heatmap(timestamps, pet)
    assert fig is not None
