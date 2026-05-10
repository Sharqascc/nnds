#!/usr/bin/env python
"""
analysis/trajectory_parser.py

Utilities to parse WorldSample(...) strings from PET CSVs
into numeric trajectories suitable for visualization.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd


def parse_worldsample_string(text: str) -> List[Tuple[float, float, float]]:
    """Parse a string containing one or more 'WorldSample(t=..., x=..., y=...)'
    patterns into a list of (t, x, y) floats."""
    if not isinstance(text, str) or not text.strip():
        return []

    matches = re.findall(
        r"t=([-\d.]+).*?x=([-\d.]+).*?y=([-\d.]+)",
        text,
    )
    return [(float(t), float(x), float(y)) for t, x, y in matches]


def parse_trajectory_column(series: pd.Series) -> pd.Series:
    """Apply parse_worldsample_string row-wise to a PET trajectory column."""
    return series.apply(parse_worldsample_string)


def attach_parsed_trajectories(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'traj_i' and 'traj_j' columns parsed from 'world_traj_i/j' if present."""
    out = df.copy()
    if "world_traj_i" in out.columns:
        out["traj_i"] = parse_trajectory_column(out["world_traj_i"])
    if "world_traj_j" in out.columns:
        out["traj_j"] = parse_trajectory_column(out["world_traj_j"])
    return out


def load_pet_with_trajectories(csv_path: Path) -> pd.DataFrame:
    """Convenience loader: read PET CSV and attach parsed trajectory lists."""
    df = pd.read_csv(csv_path)
    return attach_parsed_trajectories(df)


def save_viz_ready_pet(
    csv_path: Path,
    out_path: Optional[Path] = None,
) -> Path:
    """Load PET CSV, parse trajectories, and save a 'viz-ready' pickle."""
    if out_path is None:
        out_path = csv_path.with_suffix(".viz.pkl")

    df = load_pet_with_trajectories(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out_path)
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse WorldSample trajectories in a PET CSV.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to petevents_bev*.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output .pkl path (default: <csv>.viz.pkl)",
    )
    args = parser.parse_args()

    out = save_viz_ready_pet(args.csv, args.out)
    print(f"✅ Saved viz-ready PET file to: {out}")
