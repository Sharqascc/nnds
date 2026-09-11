#!/usr/bin/env python3
"""PET/TTC agreement report: MAE, RMSE, R^2, Spearman + scatter plot.

CSV format: track_a, track_b, pet [, ttc]
"""

from __future__ import annotations

import argparse
import json
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


def _opt_float(val):
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
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    pred = _to_events(pd.read_csv(args.predicted))
    gt = _to_events(pd.read_csv(args.ground_truth))
    out = Path(args.out_dir)

    result = {}
    for field in ("pet", "ttc"):
        m = agreement_metrics(pred, gt, field)
        print(f"{field.upper()} agreement:")
        print(f"  MAE: {m['mae']:.4f} s")
        print(f"  RMSE: {m['rmse']:.4f} s")
        print(f"  R^2: {m['r2']:.4f}")
        print(f"  Spearman rho: {m['spearman']:.4f}")
        print(f"  N: {m['n']}")
        result[f"{field}_mae_s"] = m["mae"]
        result[f"{field}_r2"] = m["r2"]
        result[f"{field}_spearman"] = m["spearman"]
        pv, gv = aligned_pairs(pred, gt, field)
        if pv.size > 0:
            path = plot_pet_agreement(
                pv, gv, out / f"{field}_agreement.png", field_label=field.upper()
            )
            print(f"  Scatter: {path}")

    if args.out_json:
        jout = Path(args.out_json)
        jout.parent.mkdir(parents=True, exist_ok=True)
        jout.write_text(json.dumps(result, indent=2, default=float))
        print(f"Wrote {jout}")


if __name__ == "__main__":
    main()
