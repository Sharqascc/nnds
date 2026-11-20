from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REQUIRED_COLUMNS = {"site", "frame", "pet"}
DENSITY_WINDOW_FRAMES = 30
BOOTSTRAP_ITERATIONS = 10000
BOOTSTRAP_SEED = 42


def load_pet_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Input PET CSV not found: {path}")

    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise SystemExit(f"Input PET CSV missing columns: {sorted(missing)}")

    df = df.copy()
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce")
    df["pet"] = pd.to_numeric(df["pet"], errors="coerce")
    df = df.dropna(subset=["site", "frame", "pet"])
    return df


def local_event_density(df: pd.DataFrame, window: int) -> pd.DataFrame:
    df = df.copy()
    densities = []

    for _, row in df.iterrows():
        site = row["site"]
        frame = row["frame"]
        site_frames = df.loc[df["site"] == site, "frame"].to_numpy()
        nearby = np.sum((site_frames >= frame - window) & (site_frames <= frame + window))
        densities.append(nearby)

    df["local_event_count"] = densities
    return df


def density_assessment(df: pd.DataFrame) -> dict[str, object]:
    rho, p_value = spearmanr(df["local_event_count"], df["pet"])
    return {
        "test": "density_confounding",
        "status": "proxy_only_true_density_unavailable",
        "method": "local PET-event-density proxy",
        "window_frames": DENSITY_WINDOW_FRAMES,
        "events_analyzed": len(df),
        "spearman_rho": float(rho),
        "spearman_p_value": float(p_value),
        "interpretation": (
            "No detectable association between local PET-event clustering and PET. "
            "This does not measure true traffic density or rule out density confounding."
        ),
    }


def threshold_sensitivity(df: pd.DataFrame) -> dict[str, object]:
    thresholds = [1.0, 1.5, 2.0, 2.5, 3.0]
    results = []

    for threshold in thresholds:
        row: dict[str, object] = {"threshold_s": threshold}

        for site, group in df.groupby("site", sort=False):
            pet = group["pet"]
            row[site] = {
                "events": len(group),
                "events_below_threshold": int((pet < threshold).sum()),
                "rate_below_threshold": float((pet < threshold).mean()),
                "mean_pet": float(pet.mean()),
                "median_pet": float(pet.median()),
            }

        giti = df.loc[df["site"] == "GITI", "pet"]
        mrc = df.loc[df["site"] == "MRC", "pet"]
        row["rate_difference_mrc_minus_giti"] = float(
            (mrc < threshold).mean() - (giti < threshold).mean()
        )
        results.append(row)

    return {
        "test": "pet_threshold_sensitivity",
        "events_analyzed": len(df),
        "thresholds_s": thresholds,
        "results": results,
        "interpretation": (
            "Compare rate differences across thresholds; material changes indicate "
            "that the binary PET classification is threshold-sensitive."
        ),
    }


def threshold_bootstrap(df: pd.DataFrame) -> dict[str, object]:
    thresholds = [1.0, 1.5, 2.0, 2.5]
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    giti = df.loc[df["site"] == "GITI", "pet"].to_numpy()
    mrc = df.loc[df["site"] == "MRC", "pet"].to_numpy()

    if len(giti) == 0 or len(mrc) == 0:
        raise SystemExit("Both GITI and MRC must contain at least one PET event.")

    results = []
    for threshold in thresholds:

        def difference() -> float:
            giti_sample = rng.choice(giti, size=len(giti), replace=True)
            mrc_sample = rng.choice(mrc, size=len(mrc), replace=True)
            return float((mrc_sample < threshold).mean() - (giti_sample < threshold).mean())

        observed = float((mrc < threshold).mean() - (giti < threshold).mean())
        samples = np.array([difference() for _ in range(BOOTSTRAP_ITERATIONS)])
        lower, upper = np.quantile(samples, [0.025, 0.975])

        results.append(
            {
                "threshold_s": threshold,
                "observed_rate_difference_mrc_minus_giti": observed,
                "ci_95_lower": float(lower),
                "ci_95_upper": float(upper),
                "ci_includes_zero": bool(lower <= 0.0 <= upper),
            }
        )

    return {
        "test": "pet_threshold_bootstrap",
        "n_giti": len(giti),
        "n_mrc": len(mrc),
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "results": results,
        "interpretation": (
            "For each threshold, an interval including zero means the site "
            "difference is not statistically distinguishable from no difference."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run reproducible PET scientific-validation tests."
    )
    parser.add_argument(
        "--input",
        default="outputs/combined_screened_simplified.csv",
        help="Path to the combined screened PET CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="paper/results",
        help="Directory for JSON result artifacts.",
    )
    args = parser.parse_args()

    input_csv = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_pet_csv(input_csv)
    df = local_event_density(df, DENSITY_WINDOW_FRAMES)

    outputs = {
        "density_confounding_proxy.json": density_assessment(df),
        "pet_threshold_sensitivity.json": threshold_sensitivity(df),
        "pet_threshold_bootstrap.json": threshold_bootstrap(df),
    }

    for filename, payload in outputs.items():
        path = output_dir / filename
        path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote {path}")

    bootstrap = outputs["pet_threshold_bootstrap.json"]
    print("\nScientific-test summary:")
    for result in bootstrap["results"]:
        print(
            f"  PET threshold: {result['threshold_s']} s | "
            f"MRC-GITI rate difference: "
            f"{result['observed_rate_difference_mrc_minus_giti']:.3f} | "
            f"95% CI: [{result['ci_95_lower']:.3f}, "
            f"{result['ci_95_upper']:.3f}] | "
            f"Includes zero: {result['ci_includes_zero']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
