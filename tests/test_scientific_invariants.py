"""
Scientific invariant tests for NNDS pipeline.
These tests verify core scientific properties of the PET event data.

NOTE: These tests require the screened output files (outputs/giti_screened.csv,
outputs/mrc_screened.csv) to be generated via `scripts/reproduce_pipeline.sh`.
If they are not present (e.g., in CI without videos), tests will be skipped.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).parent.parent


def _output_exists():
    return (REPO / "outputs/giti_screened.csv").exists() and (
        REPO / "outputs/mrc_screened.csv"
    ).exists()


def _load_results():
    giti = pd.read_csv(REPO / "outputs/giti_screened.csv")
    mrc = pd.read_csv(REPO / "outputs/mrc_screened.csv")
    # Check if files are real CSVs (not LFS pointers)
    if "pet" not in giti.columns or "pet" not in mrc.columns:
        pytest.skip("Screened outputs are LFS pointers or invalid - run reproduce_pipeline.sh")
    return giti, mrc


@pytest.mark.skipif(not _output_exists(), reason="Screened outputs not generated")
def test_pet_positive():
    giti, _ = _load_results()
    assert (giti["pet"] > 0).all()


@pytest.mark.skipif(not _output_exists(), reason="Screened outputs not generated")
def test_pet_threshold():
    giti, _ = _load_results()
    assert (giti["pet"] <= 3.0).all()


@pytest.mark.skipif(not _output_exists(), reason="Screened outputs not generated")
def test_no_same_orig():
    giti, _ = _load_results()
    assert (giti["orig_track_a"] != giti["orig_track_b"]).all()


@pytest.mark.skipif(not _output_exists(), reason="Screened outputs not generated")
def test_no_temporal_duplicates():
    """
    Same vehicle pair must not produce two events in the same grid cell
    within a short temporal window (< 10 frames / 0.33s).

    Justification: At 30 FPS, 10 frames = 0.33s. A vehicle entering a grid cell
    takes at least 10 frames to traverse and exit the cell (based on cell size
    ~100px and typical vehicle speeds). Two events separated by < 10 frames
    likely represent the same physical interaction being counted twice.
    Events separated by >= 10 frames are distinct temporal episodes where the
    vehicles interacted, left the zone, and interacted again.
    """
    giti, mrc = _load_results()
    for df, site in [(giti, "GITI"), (mrc, "MRC")]:
        key = df.apply(
            lambda r: (*tuple(sorted([r["orig_track_a"], r["orig_track_b"]])), r["grid_cell"]),
            axis=1,
        )
        for key_val in key[key.duplicated(keep=False)].unique():
            group = df[key == key_val].sort_values("frame")
            frames = group["frame"].values
            if len(frames) > 1:
                min_sep = min(np.diff(frames))
                assert min_sep >= 10, (
                    f"{site}: Duplicate (pair, grid) in {key_val[2]} with temporal separation {min_sep} frames < 10!"
                )


@pytest.mark.skipif(not _output_exists(), reason="Screened outputs not generated")
def test_world_coords():
    giti, _ = _load_results()
    for _, row in giti.iterrows():
        traj = json.loads(row["traj_a_json"])
        valid = any(p.get("world_x") is not None for p in traj)
        assert valid


@pytest.mark.skipif(not _output_exists(), reason="Screened outputs not generated")
def test_grid_cell():
    giti, _ = _load_results()
    assert not giti["grid_cell"].isin(["UNKNOWN", ""]).any()


@pytest.mark.skipif(not _output_exists(), reason="Screened outputs not generated")
def test_site():
    giti, _ = _load_results()
    assert "site" in giti.columns
    assert giti["site"].notna().all()


@pytest.mark.skipif(not _output_exists(), reason="Screened outputs not generated")
def test_mrc_invariants():
    _, mrc = _load_results()
    assert (mrc["pet"] > 0).all()
    assert (mrc["orig_track_a"] != mrc["orig_track_b"]).all()
    key = mrc.apply(
        lambda r: (*tuple(sorted([r["orig_track_a"], r["orig_track_b"]])), r["grid_cell"]), axis=1
    )
    for key_val in key[key.duplicated(keep=False)].unique():
        group = mrc[key == key_val].sort_values("frame")
        frames = group["frame"].values
        if len(frames) > 1:
            min_sep = min(np.diff(frames))
            assert min_sep >= 10, (
                f"MRC: Duplicate (pair, grid) in {key_val[2]} with temporal separation {min_sep} frames < 10!"
            )


# ------------------- Helper function coverage -------------------


def test_output_exists_false(tmp_path, monkeypatch):
    import tests.test_scientific_invariants as sci

    monkeypatch.setattr(sci, "REPO", tmp_path)
    assert not sci._output_exists()


def test_output_exists_true(tmp_path, monkeypatch):
    import tests.test_scientific_invariants as sci

    monkeypatch.setattr(sci, "REPO", tmp_path)
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "giti_screened.csv").write_text("pet\n1.0\n")
    (tmp_path / "outputs" / "mrc_screened.csv").write_text("pet\n1.0\n")
    assert sci._output_exists()


def test_load_results_valid(tmp_path, monkeypatch):
    import tests.test_scientific_invariants as sci

    monkeypatch.setattr(sci, "REPO", tmp_path)
    (tmp_path / "outputs").mkdir()
    giti_path = tmp_path / "outputs" / "giti_screened.csv"
    mrc_path = tmp_path / "outputs" / "mrc_screened.csv"
    # Create minimal valid DataFrames
    giti_df = pd.DataFrame(
        {
            "pet": [1.0],
            "orig_track_a": [1],
            "orig_track_b": [2],
            "grid_cell": ["G_A_1"],
            "frame": [10],
            "traj_a_json": ['[{"world_x": 1.0, "world_y": 2.0}]'],
            "site": ["GITI"],
        }
    )
    mrc_df = giti_df.copy()
    mrc_df["site"] = ["MRC"]
    giti_df.to_csv(giti_path, index=False)
    mrc_df.to_csv(mrc_path, index=False)
    giti, _mrc = sci._load_results()
    assert "pet" in giti.columns
    assert len(giti) == 1


def test_load_results_skip_on_missing_pet(tmp_path, monkeypatch):
    import tests.test_scientific_invariants as sci

    monkeypatch.setattr(sci, "REPO", tmp_path)
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "giti_screened.csv").write_text("foo\n1\n")
    (tmp_path / "outputs" / "mrc_screened.csv").write_text("foo\n1\n")
    with pytest.raises(pytest.skip.Exception):
        sci._load_results()
