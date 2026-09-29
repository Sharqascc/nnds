#!/usr/bin/env python3
"""
Generate deterministic natural-language descriptions for each PET event.

Reads the PET CSV (with geometric conflict_type) and outputs a Markdown report.

Usage:
    python scripts/generate_event_descriptions.py --pet-csv outputs/e2e_validation_pet.csv --output outputs/event_descriptions.md
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pet-csv", required=True)
    parser.add_argument("--output", default="outputs/event_descriptions.md")
    args = parser.parse_args()

    pet = pd.read_csv(args.pet_csv)
    lines = ["# PET Event Descriptions", ""]
    for _, row in pet.iterrows():
        lines.append(
            f"- Event {row.get('event_id')}: At frame {row.get('frame')}, "
            f"track {row.get('track_a')} and track {row.get('track_b')} "
            f"exhibited a {str(row.get('conflict_type', '')).replace('_', '-')} interaction "
            f"with PET = {row.get('pet'):.2f} s in grid cell {row.get('grid_cell', 'unknown')}."
        )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"✅ Saved event descriptions to {out_path}")


if __name__ == "__main__":
    main()
