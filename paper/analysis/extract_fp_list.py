"""Extract the 73 false positives into fp_list.csv.

Run:  python -m paper.analysis.extract_fp_list
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
REVIEWS = REPO / "data" / "reviews" / "ssm_review_114"
GITI = REPO / "outputs" / "giti_screened_with_gates.csv"
OUT = REVIEWS / "fp_list.csv"


def main() -> Path:
    labels = pd.read_csv(REVIEWS / "to_label.csv")
    mapping = pd.read_csv(REVIEWS / "label_to_giti_mapping.csv")
    giti = pd.read_csv(GITI)

    merged = labels.merge(mapping, left_on="idx", right_on="label_idx")
    merged = merged.merge(
        giti[["conflict_type", "grid_cell"]],
        left_on="giti_idx",
        right_index=True,
    )
    fps = merged[merged.verdict == "N"]
    fps[
        ["idx", "giti_idx", "track_a", "track_b", "pet", "frame", "conflict_type", "grid_cell"]
    ].to_csv(OUT, index=False)
    return OUT


if __name__ == "__main__":
    p = main()
    n = sum(1 for _ in p.open()) - 1
    print(f"wrote {p}  ({n} rows)")
