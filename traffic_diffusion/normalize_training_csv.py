import argparse
import ast
import math
import re
from pathlib import Path

import pandas as pd

WORLD_SAMPLE_RE = re.compile(
    r"WorldSample\(t=(?P<t>-?\d+(?:\.\d+)?),\s*x=(?P<x>-?\d+(?:\.\d+)?),\s*y=(?P<y>-?\d+(?:\.\d+)?)\)"
)

def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))

def parse_world_traj(value):
    if pd.isna(value):
        raise ValueError("trajectory is NaN")

    s = str(value).strip()
    if not s:
        raise ValueError("trajectory is empty")

    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list):
            out = []
            for item in obj:
                if isinstance(item, (list, tuple)) and len(item) == 3 and all(_is_num(v) for v in item):
                    t, x, y = item
                    out.append((float(t), float(x), float(y)))
                elif isinstance(item, dict) and {"t", "x", "y"} <= set(item.keys()):
                    out.append((float(item["t"]), float(item["x"]), float(item["y"])))
                else:
                    raise ValueError(f"unsupported literal item: {item!r}")
            if out:
                return out
    except Exception:
        pass

    matches = WORLD_SAMPLE_RE.findall(s)
    if matches:
        return [(float(t), float(x), float(y)) for t, x, y in matches]

    raise ValueError("unsupported trajectory format")

def normalize_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    required = ["world_traj_i", "world_traj_j"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    keep_rows = []
    dropped = 0

    for _, row in df.iterrows():
        try:
            ti = parse_world_traj(row["world_traj_i"])
            tj = parse_world_traj(row["world_traj_j"])
            row = row.copy()
            row["world_traj_i"] = repr(ti)
            row["world_traj_j"] = repr(tj)
            keep_rows.append(row)
        except Exception:
            dropped += 1

    out_df = pd.DataFrame(keep_rows, columns=df.columns)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    print({
        "input_csv": input_csv,
        "output_csv": output_csv,
        "input_rows": len(df),
        "output_rows": len(out_df),
        "dropped_rows": dropped,
    })

def main():
    p = argparse.ArgumentParser(description="Normalize PET trajectory CSV for diffusion training")
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output-csv", required=True)
    args = p.parse_args()
    normalize_csv(args.input_csv, args.output_csv)

if __name__ == "__main__":
    main()
