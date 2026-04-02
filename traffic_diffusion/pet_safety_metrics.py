
import pandas as pd

def compute_safety_metrics(df: pd.DataFrame) -> dict:
    out = {}
    out["n_events"] = int(len(df))
    out["pet_mean"] = float(df["PET"].mean())
    out["pet_median"] = float(df["PET"].median())
    out["risk_mean"] = float(df["risk_score"].mean())
    if "severity_evt" in df.columns:
        out["critical_share"] = float((df["severity_evt"] == "critical").mean())
        out["high_or_worse_share"] = float(df["severity_evt"].isin(["critical", "high"]).mean())
    else:
        out["critical_share"] = None
        out["high_or_worse_share"] = None
    out["safety_score"] = 1.0 - out["risk_mean"]
    return out
