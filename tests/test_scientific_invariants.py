"""
Scientific invariant tests for NNDS pipeline.
These tests verify core scientific properties of the PET event data.

NOTE: These tests require the screened output files (outputs/giti_screened.csv,
outputs/mrc_screened.csv) to be generated via `scripts/reproduce_pipeline.sh`.
If they are not present (e.g., in CI without videos), tests will be skipped.
"""

import pandas as pd
import json
import pytest
from pathlib import Path

REPO = Path(__file__).parent.parent

def _output_exists():
    return (REPO/'outputs/giti_screened.csv').exists() and (REPO/'outputs/mrc_screened.csv').exists()

def _load_results():
    giti = pd.read_csv(REPO/'outputs/giti_screened.csv')
    mrc = pd.read_csv(REPO/'outputs/mrc_screened.csv')
    # Check if files are real CSVs (not LFS pointers)
    if 'pet' not in giti.columns or 'pet' not in mrc.columns:
        pytest.skip("Screened outputs are LFS pointers or invalid - run reproduce_pipeline.sh")
    return giti, mrc

@pytest.mark.skipif(not _output_exists(), reason="Screened outputs not generated")
def test_pet_positive():
    giti, _ = _load_results()
    assert (giti['pet'] > 0).all()

@pytest.mark.skipif(not _output_exists(), reason="Screened outputs not generated")
def test_pet_threshold():
    giti, _ = _load_results()
    assert (giti['pet'] <= 3.0).all()

@pytest.mark.skipif(not _output_exists(), reason="Screened outputs not generated")
def test_no_same_orig():
    giti, _ = _load_results()
    assert (giti['orig_track_a'] != giti['orig_track_b']).all()

@pytest.mark.skipif(not _output_exists(), reason="Screened outputs not generated")
def test_unique_pairs():
    giti, _ = _load_results()
    pairs = giti.apply(lambda r: tuple(sorted([r['orig_track_a'], r['orig_track_b']])), axis=1)
    assert pairs.is_unique

@pytest.mark.skipif(not _output_exists(), reason="Screened outputs not generated")
def test_world_coords():
    giti, _ = _load_results()
    for _, row in giti.iterrows():
        traj = json.loads(row['traj_a_json'])
        valid = any(p.get('world_x') is not None for p in traj)
        assert valid

@pytest.mark.skipif(not _output_exists(), reason="Screened outputs not generated")
def test_grid_cell():
    giti, _ = _load_results()
    assert not giti['grid_cell'].isin(['UNKNOWN', '']).any()

@pytest.mark.skipif(not _output_exists(), reason="Screened outputs not generated")
def test_site():
    giti, _ = _load_results()
    assert 'site' in giti.columns
    assert giti['site'].notna().all()

@pytest.mark.skipif(not _output_exists(), reason="Screened outputs not generated")
def test_mrc_invariants():
    _, mrc = _load_results()
    assert (mrc['pet'] > 0).all()
    assert (mrc['orig_track_a'] != mrc['orig_track_b']).all()
