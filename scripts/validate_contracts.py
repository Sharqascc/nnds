#!/usr/bin/env python3
"""
Validate PET CSV rows against Pydantic data contracts.

Usage:
    python scripts/validate_contracts.py --csv outputs/combined_screened_simplified.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.contracts import PETEventRecord


def validate_csv(csv_path: str) -> int:
    df = pd.read_csv(csv_path)
    errors = []
    for idx, row in df.iterrows():
        row_dict = {k: (None if pd.isna(v) else v) for k, v in row.items()}
        try:
            PETEventRecord(**row_dict)
        except ValidationError as e:
            errors.append((idx, str(e)))
    if errors:
        print(f"❌ Found {len(errors)} invalid rows")
        for idx, err in errors[:10]:
            print(f"Row {idx}: {err}")
        return 1
    print(f"✅ All {len(df)} rows passed contract validation")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Validate PET CSV against contracts")
    parser.add_argument("--csv", required=True, help="Path to PET events CSV")
    args = parser.parse_args()
    sys.exit(validate_csv(args.csv))


if __name__ == "__main__":
    main()
