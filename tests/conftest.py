import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_pet_series():
    """Return a small pandas Series of PET values."""
    return pd.Series([0.5, 1.2, 2.5, 4.0, 2.0], name="pet")


@pytest.fixture
def sample_pet_df(sample_pet_series):
    """Return a DataFrame with a 'pet' column."""
    return sample_pet_series.to_frame()


@pytest.fixture
def pet_csv_path(tmp_path, sample_pet_df):
    """Create a temporary CSV with sample PET data and return its path."""
    csv_path = tmp_path / "pet_events.csv"
    sample_pet_df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def random_pet_array():
    """Return a NumPy array of random PET values (fixed seed)."""
    rng = np.random.default_rng(42)
    return rng.uniform(0.1, 5.0, size=20)


@pytest.fixture
def empty_pet_csv(tmp_path):
    """Return a path to an empty CSV (only header) to test error handling."""
    csv_path = tmp_path / "empty_pet.csv"
    pd.DataFrame(columns=["pet"]).to_csv(csv_path, index=False)
    return csv_path
