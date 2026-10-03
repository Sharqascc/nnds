"""Phase 1 statistical tests on frozen PET events.

Reproduces paper/results/stat_tests.json from outputs/*_screened_with_gates.csv.

Run:
    python -m paper.analysis.statistical_tests
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
RES = ROOT / "paper" / "results"
RES.mkdir(parents=True, exist_ok=True)

SEED = 42
N_BOOT = 10_000
SEVERITY_BINS = [-1, 1.0, 1.5, 3.0, float("inf")]
SEVERITY_LABELS = ["critical", "serious", "moderate", "safe"]


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    giti = pd.read_csv(OUT / "giti_screened_with_gates.csv")
    mrc = pd.read_csv(OUT / "mrc_screened_with_gates.csv")
    return giti, mrc


def mwu_and_ks(a: np.ndarray, b: np.ndarray) -> dict:
    u, p_u = stats.mannwhitneyu(a, b, alternative="two-sided")
    n1, n2 = len(a), len(b)
    rank_biserial = 1 - (2 * u) / (n1 * n2)
    ks, p_ks = stats.ks_2samp(a, b)
    return {
        "pet_mwu": {
            "U": float(u), "p": float(p_u),
            "rank_biserial": float(rank_biserial),
            "n_giti": n1, "n_mrc": n2,
        },
        "pet_ks": {"statistic": float(ks), "p": float(p_ks)},
    }


def bootstrap_diffs(a: np.ndarray, b: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    n1, n2 = len(a), len(b)
    d_med = np.empty(N_BOOT)
    d_mean = np.empty(N_BOOT)
    for i in range(N_BOOT):
        sa = rng.choice(a, n1, replace=True)
        sb = rng.choice(b, n2, replace=True)
        d_med[i] = np.median(sa) - np.median(sb)
        d_mean[i] = sa.mean() - sb.mean()
    return {
        "pet_median_diff_ci": {
            "point": float(np.median(a) - np.median(b)),
            "lo": float(np.percentile(d_med, 2.5)),
            "hi": float(np.percentile(d_med, 97.5)),
        },
        "pet_mean_diff_ci": {
            "point": float(a.mean() - b.mean()),
            "lo": float(np.percentile(d_mean, 2.5)),
            "hi": float(np.percentile(d_mean, 97.5)),
        },
    }


def severity_test(a: np.ndarray, b: np.ndarray) -> dict:
    sev_a = pd.cut(a, bins=SEVERITY_BINS, labels=SEVERITY_LABELS).astype(str)
    sev_b = pd.cut(b, bins=SEVERITY_BINS, labels=SEVERITY_LABELS).astype(str)
    site = pd.Series(["GITI"] * len(a) + ["MRC"] * len(b), name="site")
    sev = pd.concat([pd.Series(sev_a), pd.Series(sev_b)], ignore_index=True)
    ct = pd.crosstab(sev, site)
    ct = ct.loc[(ct.sum(axis=1) > 0), (ct.sum(axis=0) > 0)]
    chi2, p, dof, _ = stats.chi2_contingency(ct.values)
    return {
        "severity_chi2": {
            "chi2": float(chi2), "p": float(p), "dof": int(dof),
            "table": ct.to_dict(),
        }
    }


def conflict_type_test(giti: pd.DataFrame, mrc: pd.DataFrame) -> dict:
    ct = pd.crosstab(
        pd.concat([giti.conflict_type, mrc.conflict_type], ignore_index=True),
        pd.Series(["GITI"] * len(giti) + ["MRC"] * len(mrc), name="site"),
    )
    chi2, p, dof, _ = stats.chi2_contingency(ct.values)
    return {
        "conflict_type_chi2": {
            "chi2": float(chi2), "p": float(p), "dof": int(dof),
            "table": ct.to_dict(),
        }
    }


def post_hoc_power(a: np.ndarray, b: np.ndarray) -> dict:
    n1, n2 = len(a), len(b)
    pooled_sd = np.sqrt(
        ((n1 - 1) * a.std(ddof=1) ** 2 + (n2 - 1) * b.std(ddof=1) ** 2)
        / (n1 + n2 - 2)
    )
    d_cohen = (a.mean() - b.mean()) / pooled_sd
    ncp = abs(d_cohen) * np.sqrt(n1 * n2 / (n1 + n2))
    z_alpha = norm.ppf(1 - 0.05 / 2)
    power = 1 - norm.cdf(z_alpha - ncp) + norm.cdf(-z_alpha - ncp)
    return {"power": {"cohens_d": float(d_cohen),
                       "power_approx": float(power),
                       "n1": n1, "n2": n2}}


def main() -> dict:
    giti, mrc = load()
    a, b = giti.pet.values, mrc.pet.values
    results: dict = {}
    results.update(mwu_and_ks(a, b))
    results.update(bootstrap_diffs(a, b))
    results.update(severity_test(a, b))
    results.update(conflict_type_test(giti, mrc))
    results.update(post_hoc_power(a, b))
    (RES / "stat_tests.json").write_text(json.dumps(results, indent=2, default=str))
    return results


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=str))
