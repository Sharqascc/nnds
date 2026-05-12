#!/usr/bin/env python
"""
PET Event Summary & Analysis Tool - Production Grade

Provides comprehensive analysis of PET (Post-Encroachment Time) events:
- Statistical summaries with confidence intervals
- Risk assessment with configurable thresholds
- Comparison between real and generated data
- Export to JSON/CSV for reporting

Author: NNDS Team
Version: 2.0.0
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats


class PETEventAnalyzer:
    """Comprehensive PET event analyzer with statistical rigor."""

    # Default safety thresholds (seconds), inspired by traffic conflict literature.[web:37][web:39]
    DEFAULT_THRESHOLDS = {
        "critical": 1.0,   # <1s: high-risk conflict
        "serious": 2.0,    # 1–2s: likely evasive action
        "moderate": 3.0,   # 2–3s: potential conflict
        "safe": 3.0        # >3s: safe interaction
    }

    def __init__(
        self,
        csv_path: Path,
        conflict_col: str = "conflict_type",
        thresholds: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize PET analyzer.

        Args:
            csv_path: Path to PET events CSV.
            conflict_col: Column name for conflict categories.
            thresholds: Custom safety thresholds (overrides defaults).
        """
        self.csv_path = Path(csv_path)
        self.conflict_col = conflict_col
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS.copy()
        self.df: pd.DataFrame = pd.DataFrame()
        self.pet_series: pd.Series = pd.Series(dtype=float)
        self._load_and_validate()

    # ------------------------------------------------------------------ #
    # Data loading & validation
    # ------------------------------------------------------------------ #
    def _load_and_validate(self) -> None:
        """Load CSV and perform data quality checks."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)

        # Required columns
        required_cols = ["pet"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing} in {self.csv_path}")

        # Coerce PET to numeric
        original_len = len(df)
        df["pet"] = pd.to_numeric(df["pet"], errors="coerce")
        invalid_mask = df["pet"].isna()
        n_invalid = int(invalid_mask.sum())
        if n_invalid > 0:
            warnings.warn(
                f"{n_invalid} rows had invalid PET values and were removed "
                f"(from {original_len} total rows)."
            )
            df = df.loc[~invalid_mask].copy()

        if df.empty:
            raise ValueError("No valid PET values after cleaning.")

        # Range checks (domain sanity, not hard errors).
        if (df["pet"] < 0).any():
            warnings.warn("Negative PET values found; check data quality.")

        if (df["pet"] > 10).any():
            warnings.warn("Very large PET values (>10 s) found; ensure they are expected.")

        self.df = df
        self.pet_series = df["pet"]
        print(f"✅ Loaded {len(self.pet_series)} valid PET events from {self.csv_path.name}")

    # ------------------------------------------------------------------ #
    # Core statistics
    # ------------------------------------------------------------------ #
    def basic_stats(self, ci: float = 0.95) -> Dict[str, Any]:
        """
        Calculate statistical summary with confidence intervals.

        Args:
            ci: Confidence interval level (e.g. 0.95).
        """
        pet = self.pet_series
        n = int(len(pet))

        stats_dict: Dict[str, Any] = {
            "count": n,
            "mean": float(pet.mean()),
            "std": float(pet.std(ddof=1)) if n > 1 else float("nan"),
            "sem": float(pet.sem()) if n > 1 else float("nan"),
            "min": float(pet.min()),
            "q25": float(pet.quantile(0.25)),
            "median": float(pet.median()),
            "q75": float(pet.quantile(0.75)),
            "max": float(pet.max()),
            "iqr": float(pet.quantile(0.75) - pet.quantile(0.25)),
            "cv": float(pet.std(ddof=1) / pet.mean()) if pet.mean() > 0 and n > 1 else float("nan"),
            "skew": float(pet.skew()) if n > 2 else float("nan"),
            "kurtosis": float(pet.kurtosis()) if n > 3 else float("nan"),
        }

        # CI for mean (Student t).[web:38][web:40]
        if n > 1 and not np.isnan(stats_dict["sem"]):
            se = stats_dict["sem"]
            t_crit = stats.t.ppf((1 + ci) / 2.0, n - 1)
            margin = t_crit * se
            stats_dict["ci_mean_lower"] = stats_dict["mean"] - margin
            stats_dict["ci_mean_upper"] = stats_dict["mean"] + margin
            stats_dict["ci_level"] = ci

        # Percentiles for risk tail analysis.[web:37][web:39]
        for pct in (1, 5, 10, 90, 95, 99):
            stats_dict[f"p{pct}"] = float(pet.quantile(pct / 100.0))

        return stats_dict

    # ------------------------------------------------------------------ #
    # Risk assessment
    # ------------------------------------------------------------------ #
    def risk_assessment(self) -> pd.DataFrame:
        """Classify each event into risk levels based on PET thresholds."""
        crit = self.thresholds["critical"]
        mod = self.thresholds["moderate"]
        serious = self.thresholds.get("serious", (crit + mod) / 2.0)

        pet = self.pet_series

        conditions = [
            pet < crit,
            (pet >= crit) & (pet < serious),
            (pet >= serious) & (pet < mod),
            pet >= mod,
        ]
        labels = ["Critical", "Serious", "Moderate", "Safe"]

        risk_levels = np.select(conditions, labels, default="Unknown")

        return pd.DataFrame(
            {"pet": pet.values, "risk_level": risk_levels},
            index=self.df.index,
        )

    def risk_summary(self) -> Dict[str, Any]:
        """Summarize distribution of risk levels and conflict rate."""
        risk_df = self.risk_assessment()
        counts = risk_df["risk_level"].value_counts()
        total = int(len(risk_df))

        summary: Dict[str, Any] = {}
        for level in ["Critical", "Serious", "Moderate", "Safe"]:
            c = int(counts.get(level, 0))
            summary[level.lower()] = {
                "count": c,
                "percentage": float(100.0 * c / total) if total > 0 else 0.0,
            }

        conflict_count = summary["critical"]["count"] + summary["serious"]["count"]
        summary["conflict_rate"] = {
            "count": conflict_count,
            "percentage": float(100.0 * conflict_count / total) if total > 0 else 0.0,
            "per_1000_events": float(1000.0 * conflict_count / total) if total > 0 else 0.0,
        }

        return summary

    # ------------------------------------------------------------------ #
    # By-conflict-type analysis
    # ------------------------------------------------------------------ #
    def by_conflict_type(self) -> pd.DataFrame:
        """Compute PET statistics and risk rates grouped by conflict type."""
        if self.conflict_col not in self.df.columns:
            return pd.DataFrame()

        risk_df = self.risk_assessment()

        rows: List[Dict[str, Any]] = []
        for conflict_type, group in self.df.groupby(self.conflict_col):
            idx = group.index
            pet = group["pet"]
            risk_local = risk_df.loc[idx, "risk_level"]

            rows.append(
                {
                    "conflict_type": str(conflict_type),
                    "count": int(len(group)),
                    "pet_mean": float(pet.mean()),
                    "pet_std": float(pet.std(ddof=1)) if len(group) > 1 else float("nan"),
                    "pet_median": float(pet.median()),
                    "critical_rate": float((risk_local == "Critical").mean() * 100.0),
                    "serious_rate": float((risk_local == "Serious").mean() * 100.0),
                    "conflict_rate": float(
                        ((risk_local == "Critical") | (risk_local == "Serious")).mean() * 100.0
                    ),
                }
            )

        out = pd.DataFrame(rows)
        if not out.empty:
            out = out.sort_values("conflict_rate", ascending=False)
        return out

    # ------------------------------------------------------------------ #
    # Baseline comparison
    # ------------------------------------------------------------------ #
    def compare_with_baseline(self, baseline_csv: Path) -> Dict[str, Any]:
        """
        Compare this PET distribution with a baseline PET CSV.

        Uses parametric (paired t) or non‑parametric (Wilcoxon) tests depending
        on normality, plus KS test and error metrics.[web:38][web:40]
        """
        baseline = PETEventAnalyzer(baseline_csv, conflict_col=self.conflict_col)

        # Paired sample (truncate to common length).
        n = min(len(self.pet_series), len(baseline.pet_series))
        if n == 0:
            raise ValueError("No overlapping samples for comparison.")

        current_sample = self.pet_series.iloc[:n].to_numpy()
        baseline_sample = baseline.pet_series.iloc[:n].to_numpy()

        # Normality tests (D’Agostino).
        _, p_norm_cur = stats.normaltest(current_sample) if n >= 8 else (None, 0.0)
        _, p_norm_base = stats.normaltest(baseline_sample) if n >= 8 else (None, 0.0)

        if (
            p_norm_cur is not None
            and p_norm_base is not None
            and p_norm_cur > 0.05
            and p_norm_base > 0.05
        ):
            # Parametric paired t‑test
            t_stat, t_p = stats.ttest_rel(current_sample, baseline_sample)
            test_used = "paired t-test"
            effect_size = self._cohens_d(current_sample, baseline_sample)
            effect_size_type = "Cohen's d"
        else:
            # Non‑parametric Wilcoxon
            wilc_stat, wilc_p = stats.wilcoxon(current_sample, baseline_sample)
            t_stat, t_p = wilc_stat, wilc_p
            test_used = "Wilcoxon signed-rank test"
            effect_size = self._cliffs_delta(current_sample, baseline_sample)
            effect_size_type = "Cliff's delta"

        # KS test (distribution difference)
        ks_stat, ks_p = stats.ks_2samp(current_sample, baseline_sample)

        # Errors and bias
        diff = current_sample - baseline_sample
        abs_errors = np.abs(diff)
        rel_errors = abs_errors / (np.abs(baseline_sample) + 1e-8)
        var_base = float(np.var(baseline_sample, ddof=1)) if n > 1 else 0.0
        mse = float(np.mean(diff ** 2)) if n > 0 else float("nan")

        comparison: Dict[str, Any] = {
            "baseline_file": str(baseline_csv),
            "sample_size": int(n),
            "test_used": test_used,
            "test_statistic": float(t_stat),
            "p_value": float(t_p),
            "is_significant": bool(t_p < 0.05),
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_p),
            "effect_size_type": effect_size_type,
            "effect_size": float(effect_size),
            "effect_size_interpretation": self._interpret_effect_size(effect_size),
            "bias": float(np.mean(diff)) if n > 0 else float("nan"),
            "mae": float(np.mean(abs_errors)) if n > 0 else float("nan"),
            "rmse": float(np.sqrt(mse)) if n > 0 else float("nan"),
            "mape": float(np.mean(rel_errors) * 100.0) if n > 0 else float("nan"),
            "r2": float(1.0 - mse / var_base) if var_base > 0 and not np.isnan(mse) else float("nan"),
        }
        return comparison

    @staticmethod
    def _cohens_d(sample1: np.ndarray, sample2: np.ndarray) -> float:
        diff = sample1 - sample2
        sd_diff = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
        return float(np.abs(np.mean(diff)) / sd_diff) if sd_diff > 0 else 0.0

    @staticmethod
    def _cliffs_delta(sample1: np.ndarray, sample2: np.ndarray) -> float:
        """Cliff’s delta as a non‑parametric effect size."""
        n1, n2 = len(sample1), len(sample2)
        if n1 == 0 or n2 == 0:
            return 0.0
        gt = 0
        lt = 0
        for x in sample1:
            gt += np.sum(x > sample2)
            lt += np.sum(x < sample2)
        return float((gt - lt) / (n1 * n2))

    @staticmethod
    def _interpret_effect_size(d: float) -> str:
        d_abs = abs(d)
        if d_abs < 0.2:
            return "negligible"
        if d_abs < 0.5:
            return "small"
        if d_abs < 0.8:
            return "medium"
        return "large"

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #
    def export_results(
        self,
        output_dir: Path,
        baseline_csv: Optional[Path] = None,
        fmt: str = "json",
    ) -> Dict[str, Path]:
        """
        Export analysis results to files.

        Args:
            output_dir: Target directory.
            baseline_csv: Optional baseline for comparison.
            fmt: "json" or "csv" for statistics export.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        exported: Dict[str, Path] = {}

        # Statistics
        stats_dict = self.basic_stats()
        stats_file = output_dir / f"pet_statistics_{self.csv_path.stem}.{fmt}"
        if fmt == "json":
            with stats_file.open("w") as f:
                json.dump(stats_dict, f, indent=2)
        else:
            pd.DataFrame([stats_dict]).to_csv(stats_file, index=False)
        exported["statistics"] = stats_file

        # Risk summary
        risk = self.risk_summary()
        risk_file = output_dir / f"pet_risk_summary_{self.csv_path.stem}.json"
        with risk_file.open("w") as f:
            json.dump(risk, f, indent=2)
        exported["risk_summary"] = risk_file

        # Per‑event risk classification
        risk_df = self.risk_assessment()
        risk_csv = output_dir / f"pet_risk_assessment_{self.csv_path.stem}.csv"
        risk_df.to_csv(risk_csv, index=False)
        exported["risk_assessment"] = risk_csv

        # By conflict type
        by_type = self.by_conflict_type()
        if not by_type.empty:
            type_file = output_dir / f"pet_by_conflict_type_{self.csv_path.stem}.csv"
            by_type.to_csv(type_file, index=False)
            exported["by_conflict_type"] = type_file

        # Baseline comparison
        if baseline_csv is not None and Path(baseline_csv).exists():
            comp = self.compare_with_baseline(Path(baseline_csv))
            comp_file = output_dir / (
                f"pet_comparison_{self.csv_path.stem}_vs_{Path(baseline_csv).stem}.json"
            )
            with comp_file.open("w") as f:
                json.dump(comp, f, indent=2, default=str)
            exported["comparison"] = comp_file

        return exported

    # ------------------------------------------------------------------ #
    # Pretty printing
    # ------------------------------------------------------------------ #
    def print_summary(self, show_risk_buckets: bool = True) -> None:
        """Print a human‑readable summary to stdout."""
        stats_dict = self.basic_stats()
        risk = self.risk_summary()

        print("\n" + "=" * 80)
        print(f"📊 PET EVENT ANALYSIS: {self.csv_path.name}")
        print("=" * 80)

        print("\n📈 Sample statistics:")
        print(f"   Total events:        {stats_dict['count']:,}")
        print(f"   Mean PET:            {stats_dict['mean']:.3f} s")
        if "ci_mean_lower" in stats_dict:
            print(
                f"   95% CI for mean:     "
                f"[{stats_dict['ci_mean_lower']:.3f}, {stats_dict['ci_mean_upper']:.3f}]"
            )
        print(f"   Median PET:          {stats_dict['median']:.3f} s")
        print(f"   IQR:                 {stats_dict['iqr']:.3f} s")
        print(f"   Range:               [{stats_dict['min']:.3f}, {stats_dict['max']:.3f}]")
        print(f"   CV (dispersion):     {stats_dict['cv']:.3f}")

        print(
            f"\n⚠️  Risk thresholds: critical<{self.thresholds['critical']}s, "
            f"moderate<{self.thresholds['moderate']}s"
        )

        if show_risk_buckets:
            print("\n⚠️  Risk levels:")
            print(
                f"   🔴 Critical:  {risk['critical']['count']:6d} "
                f"({risk['critical']['percentage']:5.1f}%)"
            )
            print(
                f"   🟠 Serious:   {risk['serious']['count']:6d} "
                f"({risk['serious']['percentage']:5.1f}%)"
            )
            print(
                f"   🟡 Moderate:  {risk['moderate']['count']:6d} "
                f"({risk['moderate']['percentage']:5.1f}%)"
            )
            print(
                f"   🟢 Safe:      {risk['safe']['count']:6d} "
                f"({risk['safe']['percentage']:5.1f}%)"
            )
            print("\n   Total conflicts (Critical + Serious):")
            print(
                f"      {risk['conflict_rate']['count']} events "
                f"({risk['conflict_rate']['percentage']:.1f}%) "
                f"≈ {risk['conflict_rate']['per_1000_events']:.1f} per 1,000 events"
            )

        print("\n📊 PET percentiles:")
        for pct in (1, 5, 10, 25, 50, 75, 90, 95, 99):
            key = f"p{pct}"
            val = stats_dict.get(key, float("nan"))
            if not np.isnan(val):
                print(f"   {pct:2d}th: {val:.3f} s")

        by_type = self.by_conflict_type()
        if not by_type.empty:
            print(f"\n📍 Top conflict locations (by {self.conflict_col}):")
            for _, row in by_type.head(5).iterrows():
                print(
                    f"   • {row['conflict_type']}: n={row['count']}, "
                    f"PET={row['pet_mean']:.2f}s, "
                    f"Conflict Rate={row['conflict_rate']:.1f}%"
                )

        print("\n" + "=" * 80)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PET Event Summary & Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python pet_summary.py --csv-path outputs/petevents_bev_30frames.csv\n"
            "  python pet_summary.py --csv-path outputs/real_pet.csv "
            "--baseline outputs/diffusion_pet.csv --export\n"
        ),
    )

    parser.add_argument(
        "--csv-path",
        required=True,
        help="Path to PET events CSV file",
    )
    parser.add_argument(
        "--conflict-col",
        default="conflict_type",
        help="Column name for conflict categories (default: conflict_type)",
    )
    parser.add_argument(
        "--baseline",
        help="Baseline PET CSV for statistical comparison (optional)",
    )
    parser.add_argument(
        "--critical",
        type=float,
        default=1.0,
        help="Critical PET threshold in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--moderate",
        type=float,
        default=3.0,
        help="Moderate PET threshold in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export analysis results to files",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis_results",
        help="Directory for exported results (default: analysis_results)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Export format for statistics (default: json)",
    )
    parser.add_argument(
        "--no-risk-buckets",
        action="store_true",
        help="Hide detailed risk bucket printout",
    )

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    thresholds = {
        "critical": args.critical,
        "moderate": args.moderate,
        "serious": (args.critical + args.moderate) / 2.0,
        "safe": args.moderate,
    }

    analyzer = PETEventAnalyzer(
        Path(args.csv_path),
        conflict_col=args.conflict_col,
        thresholds=thresholds,
    )

    analyzer.print_summary(show_risk_buckets=not args.no_risk_buckets)
    # NOTE: If you get a syntax error here, replace `args.no-risk_buckets`
    # with `args.no_risk_buckets` (underscore), depending on your argparse name.

    if args.export:
        baseline_path = Path(args.baseline) if args.baseline else None
        exported = analyzer.export_results(
        Path(args.output_dir),
        baseline_csv=baseline_path,
        fmt=args.format,
        )
        print(f"\n✅ Exported results to {args.output_dir}/")
        for name, p in exported.items():
            print(f"   • {name}: {p}")


if __name__ == "__main__":
    main()
