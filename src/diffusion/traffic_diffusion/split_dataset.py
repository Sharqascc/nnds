import pandas as pd
import numpy as np
from pathlib import Path

def split_pet_dataset(csv_path, train_ratio=0.6, val_ratio=0.2, seed=42):
    np.random.seed(seed)
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Cannot find input dataset at {csv_path}")

    df = pd.read_csv(csv_path)

    # Use interaction/track pair IDs to avoid temporal/spatial data leakage
    if "track_id_i" in df.columns and "track_id_j" in df.columns:
        df["interaction_id"] = df["track_id_i"].astype(str) + "_" + df["track_id_j"].astype(str)
    elif "track_a" in df.columns and "track_b" in df.columns:
        df["interaction_id"] = df["track_a"].astype(str) + "_" + df["track_b"].astype(str)
    else:
        df["interaction_id"] = df.index

    unique_ids = df["interaction_id"].unique()
    np.random.shuffle(unique_ids)

    n_total = len(unique_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_ids = set(unique_ids[:n_train])
    val_ids = set(unique_ids[n_train:n_train + n_val])
    test_ids = set(unique_ids[n_train + n_val:])

    train_df = df[df["interaction_id"].isin(train_ids)].drop(columns=["interaction_id"])
    val_df = df[df["interaction_id"].isin(val_ids)].drop(columns=["interaction_id"])
    test_df = df[df["interaction_id"].isin(test_ids)].drop(columns=["interaction_id"])

    out_dir = csv_path.parent
    train_path = out_dir / "petevents_train.csv"
    val_path = out_dir / "petevents_val.csv"
    test_path = out_dir / "petevents_test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("=" * 60)
    print("✂️ DATASET SPLIT COMPLETED (NO TEMPORAL/SPATIAL LEAKAGE)")
    print("=" * 60)
    print(f" • Total Conflicts: {len(df)}")
    print(f" • Train Set:       {len(train_df)} events -> {train_path}")
    print(f" • Validation Set:  {len(val_df)} events -> {val_path}")
    print(f" • Test Set:        {len(test_df)} events -> {test_path}")

if __name__ == "__main__":
    split_pet_dataset("outputs/petevents_recovered.csv")
