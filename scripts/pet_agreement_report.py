#!/usr/bin/env python3
"""PET/TTC agreement report: MAE, RMSE, R^2, Spearman + scatter plot.

CSV format: track_a, track_b, pet [, ttc]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.ssm_agreement import (
    agreement_metrics,
    aligned_pairs,
)
from src.analysis.ssm_error import SsmEvent
from src.analysis.visualization.pet_gt_scatter import plot_pet_agreement


def _to_events(df: pd.DataFrame) -> list[SsmEvent]:
    has_pet = "pet" in df.columns
    has_ttc = "ttc" in df.columns
    a = pd.to_numeric(df["track_a"], errors="coerce").astype(int).to_numpy()
    b = pd.to_numeric(df["track_b"], errors="coerce").astype(int).to_numpy()
    events: list[SsmEvent] = []
    for i in range(len(df)):
        pet: float | None = None
        ttc: float | None = None
        if has_pet:
            v = pd.to_numeric(pd.Series([df["pet"].iloc[i]]), errors="coerce").iloc[0]
            pet = None if pd.isna(v) else float(v)
        if has_ttc:
            v = pd.to_numeric(pd.Series([df["ttc"].iloc[i]]), errors="coerce").iloc[0]
            ttc = None if pd.isna(v) else float(v)
        events.append(SsmEvent(track_a=int(a[i]), track_b=int(b[i]), pet=pet, ttc=ttc))
    return events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predicted", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()

    pred = _to_events(pd.read_csv(args.predicted))
    gt = _to_events(pd.read_csv(args.ground_truth))
    out = Path(args.out_dir)

    for field in ("pet", "ttc"):
        m = agreement_metrics(pred, gt, field)
        print(f"{field.upper()} agreement:")
        print(f"  MAE: {m['mae']:.4f} s")
        print(f"  RMSE: {m['rmse']:.4f} s")
        print(f"  R^2: {m['r2']:.4f}")
        print(f"  Spearman rho: {m['spearman']:.4f}")
        print(f"  N: {m['n']}")
        pv, gv = aligned_pairs(pred, gt, field)
        if pv.size > 0:
            path = plot_pet_agreement(
                pv, gv, out / f"{field}_agreement.png", field_label=field.upper()
            )
            print(f"  Scatter: {path}")


if __name__ == "__main__":
    main()
