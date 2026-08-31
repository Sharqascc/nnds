import pandas as pd
import json
from pathlib import Path

REPO = Path(__file__).parent.parent

def load_results():
    giti = pd.read_csv(REPO/'outputs/giti_screened.csv')
    mrc = pd.read_csv(REPO/'outputs/mrc_screened.csv')
    return giti, mrc

def test_pet_positive():
    giti, _ = load_results()
    assert (giti['pet'] > 0).all()

def test_pet_threshold():
    giti, _ = load_results()
    assert (giti['pet'] <= 3.0).all()

def test_no_same_orig():
    giti, _ = load_results()
    assert (giti['orig_track_a'] != giti['orig_track_b']).all()

def test_unique_pairs():
    giti, _ = load_results()
    pairs = giti.apply(lambda r: tuple(sorted([r['orig_track_a'], r['orig_track_b']])), axis=1)
    assert pairs.is_unique

def test_world_coords():
    giti, _ = load_results()
    for _, row in giti.iterrows():
        traj = json.loads(row['traj_a_json'])
        valid = any(p.get('world_x') is not None for p in traj)
        assert valid

def test_grid_cell():
    giti, _ = load_results()
    assert not giti['grid_cell'].isin(['UNKNOWN', '']).any()

def test_site():
    giti, _ = load_results()
    assert 'site' in giti.columns
    assert giti['site'].notna().all()

def test_mrc_invariants():
    _, mrc = load_results()
    assert (mrc['pet'] > 0).all()
    assert (mrc['orig_track_a'] != mrc['orig_track_b']).all()
