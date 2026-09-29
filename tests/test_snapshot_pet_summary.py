import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.analysis.pet_summary import PETEventAnalyzer


def _round_floats(obj, ndigits=6):
    """Round floats recursively so snapshots are stable across BLAS/numpy builds.

    Raw np.float64 values can differ in the last ~1e-13 between CI and local
    because of summation-order differences in BLAS-backed reductions. The
    underlying statistics are unchanged; only their representation drifts.
    """
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_round_floats(v, ndigits) for v in obj)
    if isinstance(obj, (float, np.floating)):
        return round(float(obj), ndigits)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    return obj


def test_pet_summary_snapshot(snapshot, tmp_path):
    # Create small PET CSV
    csv = tmp_path / "pet.csv"
    pd.DataFrame(
        {
            "pet": [0.5, 1.2, 2.5, 4.0],
            "conflict_type": ["crossing", "head_on", "rear_end", "side_swipe"],
        }
    ).to_csv(csv, index=False)

    analyzer = PETEventAnalyzer(csv)
    stats = analyzer.basic_stats()
    # Snapshot the statistics dict with floats rounded to 6 dp so the value is
    # reproducible across environments.
    assert _round_floats(stats) == snapshot
