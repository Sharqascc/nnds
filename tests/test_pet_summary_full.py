from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.analysis.pet_summary import PETEventAnalyzer, main, parse_args


def make_pet_csv(tmp_path, data=None, conflict_col="conflict_type"):
    """Create a minimal PET CSV file."""
    if data is None:
        data = {
            "pet": [0.5, 1.2, 2.5, 4.0, 2.0],
            "conflict_type": ["crossing", "head_on", "rear_end", "crossing", "side_swipe"],
        }
    df = pd.DataFrame(data)
    path = tmp_path / "pet_events.csv"
    df.to_csv(path, index=False)
    return path


def test_constructor_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        PETEventAnalyzer(tmp_path / "missing.csv")


def test_constructor_missing_required_col(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("foo,bar\n1,2\n")
    with pytest.raises(ValueError):
        PETEventAnalyzer(path)


def test_constructor_invalid_pet_values(tmp_path):
    path = tmp_path / "invalid.csv"
    path.write_text("pet\n1\nnotanumber\n2\n")
    with pytest.warns(UserWarning):
        analyzer = PETEventAnalyzer(path)
    assert len(analyzer.pet_series) == 2


def test_constructor_all_invalid(tmp_path):
    path = tmp_path / "all_invalid.csv"
    path.write_text("pet\nnotanumber\nfoo\n")
    with pytest.raises(ValueError):
        PETEventAnalyzer(path)


def test_constructor_negative_and_large(tmp_path):
    path = tmp_path / "warn.csv"
    path.write_text("pet\n-0.5\n12.0\n")
    with pytest.warns(UserWarning):
        analyzer = PETEventAnalyzer(path)
    assert len(analyzer.pet_series) == 2


def test_basic_stats_small_n(tmp_path):
    path = make_pet_csv(tmp_path, data={"pet": [1.0]})
    analyzer = PETEventAnalyzer(path)
    stats = analyzer.basic_stats()
    assert stats["count"] == 1
    assert np.isnan(stats["std"])
    assert "ci_mean_lower" not in stats
    # percentiles should exist
    assert "p1" in stats
    assert "p99" in stats


def test_basic_stats_large_n(tmp_path):
    data = {"pet": np.linspace(0.5, 5.0, 20)}
    path = make_pet_csv(tmp_path, data=data)
    analyzer = PETEventAnalyzer(path)
    stats = analyzer.basic_stats(ci=0.95)
    assert stats["count"] == 20
    assert "ci_mean_lower" in stats
    assert "ci_mean_upper" in stats
    assert "cv" in stats
    assert "skew" in stats
    assert "kurtosis" in stats


def test_risk_assessment_and_summary(tmp_path):
    path = make_pet_csv(tmp_path, data={"pet": [0.5, 1.5, 2.5, 5.0]})
    analyzer = PETEventAnalyzer(path)
    risk_df = analyzer.risk_assessment()
    assert list(risk_df["risk_level"]) == ["Critical", "Serious", "Moderate", "Safe"]
    summary = analyzer.risk_summary()
    assert summary["critical"]["count"] == 1
    assert summary["safe"]["count"] == 1
    assert summary["conflict_rate"]["count"] == 2
    assert summary["conflict_rate"]["per_1000_events"] == 500.0


def test_by_conflict_type_missing_column(tmp_path):
    path = make_pet_csv(tmp_path, data={"pet": [1, 2]})
    analyzer = PETEventAnalyzer(path, conflict_col="non_existent")
    assert analyzer.by_conflict_type().empty


def test_by_conflict_type_present(tmp_path):
    path = make_pet_csv(tmp_path)
    analyzer = PETEventAnalyzer(path)
    by_type = analyzer.by_conflict_type()
    assert "conflict_type" in by_type.columns
    assert "conflict_rate" in by_type.columns
    # sorted by conflict_rate descending
    assert by_type["conflict_rate"].is_monotonic_decreasing


def test_compare_with_baseline_no_overlap(tmp_path):
    make_pet_csv(tmp_path, data={"pet": []})  # empty
    make_pet_csv(tmp_path, data={"pet": []})
    # Need analyzer with empty series? _load_and_validate raises ValueError for empty.
    # So create non-empty and then monkeypatch to empty series? Better use two non-empty with n>0.
    # Actually n=min(len) so both non-empty but maybe zero after truncate? No.
    # We'll create non-empty but then force pet_series to empty list via monkeypatch.
    analyzer = PETEventAnalyzer(make_pet_csv(tmp_path, data={"pet": [1, 2]}))
    baseline = PETEventAnalyzer(make_pet_csv(tmp_path, data={"pet": [1, 2]}))
    analyzer.pet_series = pd.Series(dtype=float)
    baseline.pet_series = pd.Series(dtype=float)
    with pytest.raises(ValueError):
        analyzer.compare_with_baseline(make_pet_csv(tmp_path, data={"pet": [1, 2]}))


def test_compare_with_baseline_parametric(tmp_path):
    path1 = make_pet_csv(tmp_path, data={"pet": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
    path2 = make_pet_csv(tmp_path, data={"pet": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1]})
    analyzer = PETEventAnalyzer(path1)
    with (
        patch("scipy.stats.normaltest", return_value=(1.0, 0.1)),
        patch("scipy.stats.ttest_rel", return_value=(0.5, 0.04)),
        patch("scipy.stats.ks_2samp", return_value=(0.2, 0.3)),
    ):
        result = analyzer.compare_with_baseline(path2)
    assert result["test_used"] == "paired t-test"
    assert result["is_significant"]
    assert "r2" in result


def test_compare_with_baseline_nonparametric(tmp_path):
    path1 = make_pet_csv(tmp_path, data={"pet": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
    path2 = make_pet_csv(tmp_path, data={"pet": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1]})
    analyzer = PETEventAnalyzer(path1)
    with (
        patch("scipy.stats.normaltest", return_value=(None, 0.01)),
        patch("scipy.stats.wilcoxon", return_value=(2.0, 0.03)),
        patch("scipy.stats.ks_2samp", return_value=(0.25, 0.4)),
    ):
        result = analyzer.compare_with_baseline(path2)
    assert result["test_used"] == "Wilcoxon signed-rank test"
    assert result["effect_size_type"] == "Cliff's delta"


def test_cohens_d():
    a = np.array([1, 2, 3])
    b = np.array([1, 2, 3])
    assert PETEventAnalyzer._cohens_d(a, b) == 0.0
    a = np.array([1, 2, 3])
    b = np.array([4, 6, 8])
    d = PETEventAnalyzer._cohens_d(a, b)
    assert d > 0


def test_cliffs_delta():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    d = PETEventAnalyzer._cliffs_delta(a, b)
    assert d == -1.0  # all a < b


def test_interpret_effect_size():
    assert PETEventAnalyzer._interpret_effect_size(0.1) == "negligible"
    assert PETEventAnalyzer._interpret_effect_size(0.3) == "small"
    assert PETEventAnalyzer._interpret_effect_size(0.6) == "medium"
    assert PETEventAnalyzer._interpret_effect_size(0.9) == "large"


def test_export_results_json(tmp_path):
    path = make_pet_csv(tmp_path)
    analyzer = PETEventAnalyzer(path)
    out_dir = tmp_path / "export"
    exported = analyzer.export_results(out_dir, fmt="json")
    assert (out_dir / f"pet_statistics_{path.stem}.json").exists()
    assert (out_dir / f"pet_risk_summary_{path.stem}.json").exists()
    assert (out_dir / f"pet_risk_assessment_{path.stem}.csv").exists()
    assert (out_dir / f"pet_by_conflict_type_{path.stem}.csv").exists()
    # no baseline comparison
    assert "comparison" not in exported


def test_export_results_csv_and_baseline(tmp_path):
    path = make_pet_csv(tmp_path)
    baseline_path = make_pet_csv(tmp_path, data={"pet": [1.0, 2.0, 3.0, 4.0, 5.0]})
    analyzer = PETEventAnalyzer(path)
    out_dir = tmp_path / "export2"
    with patch(
        "src.analysis.pet_summary.PETEventAnalyzer.compare_with_baseline", return_value={"dummy": 1}
    ):
        exported = analyzer.export_results(out_dir, baseline_csv=baseline_path, fmt="csv")
    assert (out_dir / f"pet_statistics_{path.stem}.csv").exists()
    assert "comparison" in exported


def test_print_summary(capsys, tmp_path):
    path = make_pet_csv(tmp_path)
    analyzer = PETEventAnalyzer(path)
    analyzer.print_summary(show_risk_buckets=True)
    captured = capsys.readouterr()
    assert "PET EVENT ANALYSIS" in captured.out
    assert "Critical" in captured.out


def test_parse_args_required():
    with pytest.raises(SystemExit):
        parse_args([])


def test_parse_args_defaults():
    args = parse_args(["--csv-path", "dummy.csv"])
    assert args.conflict_col == "conflict_type"
    assert args.critical == 1.0
    assert args.moderate == 3.0
    assert args.format == "json"


def test_main_export_and_print(tmp_path, capsys):
    path = make_pet_csv(tmp_path)
    out_dir = tmp_path / "main_export"
    main(["--csv-path", str(path), "--export", "--output-dir", str(out_dir)])
    assert (out_dir / f"pet_statistics_{path.stem}.json").exists()


def test_main_no_export(tmp_path, capsys):
    path = make_pet_csv(tmp_path)
    main(["--csv-path", str(path)])
    captured = capsys.readouterr()
    assert "PET EVENT ANALYSIS" in captured.out


def test_cliffs_delta_empty():
    assert PETEventAnalyzer._cliffs_delta(np.array([]), np.array([1, 2])) == 0.0
    assert PETEventAnalyzer._cliffs_delta(np.array([1, 2]), np.array([])) == 0.0
