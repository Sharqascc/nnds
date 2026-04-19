import argparse
from pathlib import Path

import pandas as pd


def summarize_pet(csv_path: str) -> None:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)

    if "pet" not in df.columns:
        raise ValueError(f"'pet' column not found in {path}")

    print(f"=== PET summary for {path} ===")

    print("\nPET (seconds) describe:")
    print(df["pet"].describe())

    if "conflict_type" in df.columns:
        print("\nCounts by conflict_type (grid cell):")
        print(df["conflict_type"].value_counts())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize PET CSV")
    parser.add_argument(
        "--csv-path",
        default="outputs/petevents_bev.csv",
        help="PET events CSV path (default: outputs/petevents_bev.csv)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summarize_pet(args.csv_path)
