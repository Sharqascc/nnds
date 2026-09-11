#!/usr/bin/env python3
"""Emit the 15-row gold-standard table from pre-computed metric JSON files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.gold_standard import build_table, to_dicts, to_markdown


def _load(path: str | None) -> dict[str, float]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detection-metrics", default=None)
    parser.add_argument("--tracking-metrics", default=None)
    parser.add_argument("--trajectory-metrics", default=None)
    parser.add_argument("--ssm-metrics", default=None)
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    rows = build_table(
        detection=_load(args.detection_metrics),
        tracking=_load(args.tracking_metrics),
        trajectory=_load(args.trajectory_metrics),
        ssm=_load(args.ssm_metrics),
    )
    md = to_markdown(rows)
    print(md)

    if args.out_md:
        out = Path(args.out_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md + "\n")
        print(f"\nWrote {out}")

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(to_dicts(rows), indent=2))
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
