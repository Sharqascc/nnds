import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.analysis.pet_summary import PETEventAnalyzer


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
    # Snapshot the statistics dict
    assert stats == snapshot
