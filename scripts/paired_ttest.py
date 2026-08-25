#!/usr/bin/env python3
"""
Perform paired t-test on PET values from two output CSV files.

Usage:
    python scripts/paired_ttest.py --file1 outputs/method1.csv --file2 outputs/method2.csv
"""
import argparse
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file1", required=True)
    parser.add_argument("--file2", required=True)
    parser.add_argument("--pet-column", default="pet")
    args = parser.parse_args()

    df1 = pd.read_csv(args.file1)
    df2 = pd.read_csv(args.file2)

    if args.pet_column not in df1.columns or args.pet_column not in df2.columns:
        raise ValueError(f"Column '{args.pet_column}' missing in one of the files")

    # Align by event_id if present; otherwise use index
    if 'event_id' in df1.columns and 'event_id' in df2.columns:
        merged = pd.merge(df1[[ 'event_id', args.pet_column]], df2[['event_id', args.pet_column]],
                          on='event_id', suffixes=('_1', '_2'))
        sample1 = merged[f'{args.pet_column}_1'].values
        sample2 = merged[f'{args.pet_column}_2'].values
    else:
        sample1 = df1[args.pet_column].values
        sample2 = df2[args.pet_column].values

    if len(sample1) < 2:
        print("Not enough paired samples to perform t-test.")
        return

    t_stat, p_value = ttest_rel(sample1, sample2)
    # Effect size: Cohen's d for paired samples
    diff = sample1 - sample2
    d = np.mean(diff) / np.std(diff, ddof=1) if len(diff) > 1 else 0.0
    print(f"Paired t-test between {args.file1} and {args.file2}")
    print(f"  Number of pairs: {len(sample1)}")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Effect size (Cohen's d): {d:.4f}")
    if p_value < 0.05:
        print("  Result: Statistically significant difference (p < 0.05)")
    else:
        print("  Result: No statistically significant difference")

if __name__ == "__main__":
    main()
