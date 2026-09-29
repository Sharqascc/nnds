#!/usr/bin/env python3
"""
Generate a summary results table comparing methods.

Usage: python scripts/generate_results_table.py
"""

import os

import pandas as pd


def main():
    data = {
        "Method": [
            "Constant Velocity",
            "Constant Acceleration",
            "Kalman Filter",
            "Social Force",
            "Diffusion (proposed)",
        ],
        "ADE": [0.0, 0.0, 0.0, 0.0, 0.0],
        "FDE": [0.0, 0.0, 0.0, 0.0, 0.0],
        "PET MAE": [0.0, 0.0, 0.0, 0.0, 0.0],
        "PET RMSE": [0.0, 0.0, 0.0, 0.0, 0.0],
    }
    df = pd.DataFrame(data)

    # Ensure the output directory exists
    os.makedirs("outputs", exist_ok=True)

    df.to_csv("outputs/results_table.csv", index=False)
    print(df.to_string(index=False))
    print("\nResults table saved to outputs/results_table.csv")


if __name__ == "__main__":
    main()
