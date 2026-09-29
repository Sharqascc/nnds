#!/usr/bin/env python3
"""Print track stability diagnostics from a detection CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.track_stability import full_report, tracks_from_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections-csv", required=True)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.detections_csv)
    required = {"frame", "track_id", "x1", "y1", "x2", "y2"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"missing columns: {sorted(missing)}")

    df["w"] = df["x2"] - df["x1"]
    df["h"] = df["y2"] - df["y1"]
    if "cx" not in df.columns:
        df["cx"] = (df["x1"] + df["x2"]) / 2.0
    if "cy" not in df.columns:
        df["cy"] = (df["y1"] + df["y2"]) / 2.0

    tracks = tracks_from_rows(df.to_dict(orient="records"))
    report = full_report(tracks)

    for section, values in report.items():
        print(f"=== {section} ===")
        for k, v in values.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        print()

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=float))
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
