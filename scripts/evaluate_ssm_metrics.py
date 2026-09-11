#!/usr/bin/env python3
"""PET / TTC value accuracy vs GT.

CSV format: track_a, track_b, pet [, ttc]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.ssm_error import SsmEvent, ssm_value_metrics


def _opt_float(val: object) -> float | None:
    if val is None:
        return None
    f = pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]
    return None if pd.isna(f) else float(f)


def _to_events(df: pd.DataFrame) -> list[SsmEvent]:
    a = pd.to_numeric(df["track_a"], errors="coerce").astype(int).to_numpy()
    b = pd.to_numeric(df["track_b"], errors="coerce").astype(int).to_numpy()
    has_pet = "pet" in df.columns
    has_ttc = "ttc" in df.columns
    events: list[SsmEvent] = []
    for i in range(len(df)):
        pet = _opt_float(df["pet"].iloc[i]) if has_pet else None
        ttc = _opt_float(df["ttc"].iloc[i]) if has_ttc else None
        events.append(SsmEvent(track_a=int(a[i]), track_b=int(b[i]), pet=pet, ttc=ttc))
    return events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predicted", required=True)
    parser.add_argument("--ground-truth", required=True)
    args = parser.parse_args()

    pred = _to_events(pd.read_csv(args.predicted))
    gt = _to_events(pd.read_csv(args.ground_truth))
    metrics = ssm_value_metrics(pred, gt)

    for field in ("pet", "ttc"):
        m = metrics[field]
        print(f"{field.upper()} value metrics:")
        print(f"  MAE: {m['mae']:.4f} s")
        print(f"  RMSE: {m['rmse']:.4f} s")
        print(f"  Matched pairs: {m['n_matched']}")
        print(f"  Pred-only (FP): {m['n_pred_only']}")
        print(f"  GT-only (FN): {m['n_gt_only']}")


if __name__ == "__main__":
    main()
