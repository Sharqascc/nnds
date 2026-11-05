"""Ground-truth PET validation (Phase 3).

Consumes workbook_annotator_{A,B}.csv, computes Cohen's kappa,
precision, recall, F1 vs the pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[2]
ANN = ROOT / "paper" / "annotations"
RES = ROOT / "paper" / "results"
RES.mkdir(parents=True, exist_ok=True)


def main() -> dict:
    a = pd.read_csv(ANN / "workbook_annotator_A.csv")
    b = pd.read_csv(ANN / "workbook_annotator_B.csv")
    m = a[["item_id", "kind", "judged_event"]].merge(
        b[["item_id", "judged_event"]], on="item_id", suffixes=("_A", "_B")
    )

    y_pipeline = (m["kind"] == "event").astype(int).values
    y_A = (m["judged_event_A"].str.lower() == "yes").astype(int).values
    y_B = (m["judged_event_B"].str.lower() == "yes").astype(int).values
    agree = y_A == y_B
    y_cons = y_A[agree]

    results: dict = {
        "n_items": len(m),
        "n_events_sampled": int((m["kind"] == "event").sum()),
        "n_controls_sampled": int((m["kind"] == "control").sum()),
        "annotator_agreement_items": int(agree.sum()),
        "cohen_kappa": float(cohen_kappa_score(y_A, y_B)),
    }
    if agree.sum() > 0:
        yp = y_pipeline[agree]
        results.update(
            {
                "precision": float(precision_score(y_cons, yp, zero_division=0)),
                "recall": float(recall_score(y_cons, yp, zero_division=0)),
                "f1": float(f1_score(y_cons, yp, zero_division=0)),
            }
        )
    (RES / "gt_validation.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
