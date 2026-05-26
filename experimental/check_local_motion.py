from pathlib import Path
import argparse
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--origin-mode", choices=["first", "min"], default="first")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    need = {"frame_idx", "track_id", "world_x", "world_y"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns: {sorted(miss)}")

    base = df[["world_x", "world_y"]].dropna()
    if base.empty:
        raise ValueError("No valid world coordinates found")

    if args.origin_mode == "first":
        ref = base.iloc[0]
    else:
        ref = pd.Series({"world_x": base["world_x"].min(), "world_y": base["world_y"].min()})

    df["world_x_local"] = df["world_x"] - ref["world_x"]
    df["world_y_local"] = df["world_y"] - ref["world_y"]

    rows = []
    for tid, g in df.groupby("track_id"):
        g = g.sort_values("frame_idx").copy()
        if len(g) < 3:
            continue
        g["dx"] = g["world_x_local"].diff()
        g["dy"] = g["world_y_local"].diff()
        g["dt"] = g["frame_idx"].diff() / args.fps
        g["step_m"] = np.sqrt(g["dx"]**2 + g["dy"]**2)
        g["speed_mps"] = g["step_m"] / g["dt"]
        valid = g["speed_mps"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) == 0:
            continue
        rows.append({
            "track_id": tid,
            "n_frames": int(len(g)),
            "median_step_m": float(g["step_m"].median()),
            "max_step_m": float(g["step_m"].max()),
            "median_speed_mps": float(valid.median()),
            "p95_speed_mps": float(valid.quantile(0.95)),
            "max_speed_mps": float(valid.max()),
            "median_speed_kmh": float((valid * 3.6).median()),
            "unstable_track": bool(valid.quantile(0.95) > 15 or g["step_m"].max() > 1.5),
        })

    out_df = pd.DataFrame(rows).sort_values(
        ["unstable_track", "median_speed_mps"], ascending=[False, False]
    )

    print("reference_origin:", ref.to_dict())
    print(out_df.head(20).to_string(index=False))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.out, index=False)
        print(f"saved_summary={args.out}")

if __name__ == "__main__":
    main()
