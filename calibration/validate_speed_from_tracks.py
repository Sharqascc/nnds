import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

def pixel_to_world(H, x, y):
    pts = np.float32([[[x, y]]])
    out = cv2.perspectiveTransform(pts, H)[0, 0]
    return float(out[0]), float(out[1])

def estimate_speed_kmh(track, H, fps):
    if len(track) < 2:
        return np.nan, [], []

    frames = np.array([p["frame"] for p in track], dtype=float)
    xy = np.array([[p["x"], p["y"]] for p in track], dtype=float)
    world = np.array([pixel_to_world(H, x, y) for x, y in xy], dtype=float)

    dists_m = np.linalg.norm(world[1:] - world[:-1], axis=1)
    dt_s = (frames[1:] - frames[:-1]) / float(fps)

    good = np.isfinite(dists_m) & np.isfinite(dt_s) & (dt_s > 0)
    if good.sum() == 0:
        return np.nan, dists_m.tolist(), dt_s.tolist()

    inst_kmh = dists_m[good] / dt_s[good] * 3.6
    inst_kmh = inst_kmh[np.isfinite(inst_kmh)]

    if len(inst_kmh) == 0:
        return np.nan, dists_m.tolist(), dt_s.tolist()

    return float(np.median(inst_kmh)), dists_m.tolist(), dt_s.tolist()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--homography-npy", required=True)
    ap.add_argument("--tracks-json", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    H = np.load(args.homography_npy)
    if H.shape != (3, 3):
        raise ValueError(f"Homography must be 3x3, got {H.shape}")

    payload = json.load(open(args.tracks_json, "r"))
    tracks = payload["tracks"] if isinstance(payload, dict) and "tracks" in payload else payload

    rows = []
    for item in tracks:
        fps = float(item["fps"])
        ref = float(item["reference_speed_kmh"])
        est, dists_m, dt_s = estimate_speed_kmh(item["track"], H, fps)

        err = est - ref if np.isfinite(est) else np.nan
        rows.append({
            "video_id": item.get("video_id"),
            "vehicle_id": item.get("vehicle_id"),
            "fps": fps,
            "n_track_points": len(item.get("track", [])),
            "reference_speed_kmh": ref,
            "estimated_speed_kmh": est,
            "error_kmh": err,
            "abs_error_kmh": abs(err) if np.isfinite(err) else np.nan,
            "pct_error": (abs(err) / ref * 100.0) if ref and np.isfinite(err) else np.nan
        })

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "per_track.csv", index=False)

    valid_err = df["error_kmh"].dropna().to_numpy()
    valid_abs = df["abs_error_kmh"].dropna().to_numpy()

    summary = {
        "count_total": int(len(df)),
        "count_valid": int(np.isfinite(df["estimated_speed_kmh"]).sum()),
        "mae_kmh": float(valid_abs.mean()) if len(valid_abs) else None,
        "rmse_kmh": float(np.sqrt(np.mean(valid_err ** 2))) if len(valid_err) else None,
        "bias_kmh": float(valid_err.mean()) if len(valid_err) else None
    }

    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
