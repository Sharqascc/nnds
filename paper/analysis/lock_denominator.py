"""Lock the precision denominator and write paper/results/pet_audit.json.

Run:  python -m paper.analysis.lock_denominator
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
REVIEWS = REPO / "data" / "reviews" / "ssm_review_114"
OUT = REPO / "paper" / "results" / "pet_audit.json"


def main() -> Path:
    labels = pd.read_csv(REVIEWS / "to_label.csv")
    mapping = pd.read_csv(REVIEWS / "label_to_giti_mapping.csv")

    matched_idx = set(mapping.label_idx)
    matched = labels[labels.idx.isin(matched_idx)]
    unmatched = labels[~labels.idx.isin(matched_idx)]

    mY = int((matched.verdict == "Y").sum())
    mN = int((matched.verdict == "N").sum())
    uY = int((unmatched.verdict == "Y").sum())
    uN = int((unmatched.verdict == "N").sum())

    audit = {
        "source": "data/reviews/ssm_review_114/to_label.csv",
        "n_audited_total": len(labels),
        "n_measurable": len(matched),
        "n_subresolution": len(unmatched),
        "pet_resolution_threshold_s": 0.10,
        "resolution_rule": "PET < 0.10 s = fewer than 3 frames at 30 fps; not reliably measurable",
        "matched_genuine": mY,
        "matched_spurious": mN,
        "precision_measurable": mY / (mY + mN) if (mY + mN) else 0.0,
        "unmatched_genuine": uY,
        "unmatched_spurious": uN,
        "precision_all_candidates": mY / (mY + mN + uY + uN),
        "subresolution_note": (
            f"{uY} of {len(unmatched)} sub-resolution events were confirmed as genuine conflicts. "
            "The pipeline detects these but cannot report their PET accurately. "
            "This is a measurement floor, not a false-positive class."
        ),
        "selection_rule": (
            "The 114 audited events were the first 114 rows of giti_screened.csv "
            "in original (pre-screen) order. After the later PET < 0.10 s screen, "
            "103 remain and map monotonically to current rows 1..152. "
            "This is a consecutive temporal window, not a random sample."
        ),
        "selection_rule_limitation": (
            "The audited 114 events are a consecutive block, not a random sample. "
            "They likely cover an early portion of the video rather than the full "
            "recording. Precision on this block may differ from precision on the "
            "remaining 50 events. A random re-sample is required to test for "
            "sampling bias and is proposed as future work."
        ),
    }
    OUT.write_text(json.dumps(audit, indent=2))
    return OUT


if __name__ == "__main__":
    p = main()
    print(f"wrote {p}  ({p.stat().st_size} bytes)")
