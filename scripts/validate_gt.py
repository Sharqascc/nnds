#!/usr/bin/env python3
"""Validate ground-truth CSV files. Exits non-zero if any errors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.gt_validation import (
    has_errors,
    summarise,
    validate_detection_rows,
    validate_ssm_rows,
    validate_tracking_rows,
    validate_trajectory_rows,
)


def _load(path: str | None) -> list[dict]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        print(f"ERROR: file not found: {path}")
        return []
    df = pd.read_csv(p)
    return df.to_dict(orient="records")


def _report(name: str, issues) -> bool:
    s = summarise(issues)
    status = "FAIL" if s["error"] else ("WARN" if s["warning"] else "OK")
    print(f"[{status}] {name}: {s['error']} error(s), {s['warning']} warning(s)")
    for i in issues[:20]:
        loc = f" row {i.row}" if i.row is not None else ""
        print(f"    {i.severity}: {i.kind}{loc}: {i.message}")
    if len(issues) > 20:
        print(f"    ... and {len(issues) - 20} more")
    return s["error"] == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detection", default=None)
    parser.add_argument("--tracking", default=None)
    parser.add_argument("--trajectory", default=None)
    parser.add_argument("--ssm", default=None)
    args = parser.parse_args()

    if not any([args.detection, args.tracking, args.trajectory, args.ssm]):
        parser.error("provide at least one --<type> path")

    ok = True

    track_ids: set[int] | None = None
    if args.tracking:
        trk_rows = _load(args.tracking)
        ok &= _report("tracking", validate_tracking_rows(trk_rows))
        track_ids = {int(r["track_id"]) for r in trk_rows if "track_id" in r}

    if args.detection:
        ok &= _report("detection", validate_detection_rows(_load(args.detection)))

    if args.trajectory:
        ok &= _report("trajectory", validate_trajectory_rows(_load(args.trajectory)))

    if args.ssm:
        ok &= _report(
            "ssm",
            validate_ssm_rows(_load(args.ssm), known_track_ids=track_ids),
        )

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
