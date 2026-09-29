import subprocess
import sys
from pathlib import Path

import pandas as pd

repo = Path(__file__).resolve().parents[1]


def test_paired_ttest_script(tmp_path):
    # Create two CSV files with different PET values
    df1 = pd.DataFrame({"event_id": [0, 1, 2], "pet": [1.0, 1.2, 1.4]})
    df2 = pd.DataFrame({"event_id": [0, 1, 2], "pet": [1.5, 1.7, 1.9]})
    f1 = tmp_path / "method1.csv"
    f2 = tmp_path / "method2.csv"
    df1.to_csv(f1, index=False)
    df2.to_csv(f2, index=False)

    result = subprocess.run(
        [sys.executable, "scripts/paired_ttest.py", "--file1", str(f1), "--file2", str(f2)],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Paired t-test" in result.stdout
