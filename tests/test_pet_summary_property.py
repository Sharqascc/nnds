
import tempfile
from pathlib import Path

import pandas as pd
from hypothesis import given
from hypothesis import strategies as st

from src.analysis.pet_summary import PETEventAnalyzer


def _make_analyzer(pets, conflict_types=None):
    """Create a PETEventAnalyzer backed by a fresh temp CSV."""
    df = pd.DataFrame({"pet_s": pets})
    if conflict_types is not None:
        df["conflict_type"] = conflict_types
    tmpdir = tempfile.mkdtemp()
    csv_path = Path(tmpdir) / "pet.csv"
    df.to_csv(csv_path, index=False)
    return PETEventAnalyzer(csv_path)


# ------------------------------------------------------------------ #
# basic_stats
# ------------------------------------------------------------------ #
@given(
    st.lists(
        st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=30,
    )
)
def test_basic_stats_quantile_ordering(pets):
    analyzer = _make_analyzer(pets)
    s = analyzer.basic_stats()

    assert s["count"] == len(pets)
    tol = 1e-9
    assert s["min"] - tol <= s["q25"] <= s["median"] <= s["q75"] <= s["max"] + tol
    assert s["min"] - tol <= s["mean"] <= s["max"] + tol
    assert abs(s["iqr"] - (s["q75"] - s["q25"])) < 1e-9


@given(
    st.lists(
        st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=3,
        max_size=30,
    )
)
def test_basic_stats_ci_brackets_mean(pets):
    analyzer = _make_analyzer(pets)
    s = analyzer.basic_stats(ci=0.95)

    assert s["ci_mean_lower"] <= s["mean"] <= s["ci_mean_upper"]
    assert s["ci_level"] == 0.95


@given(
    st.lists(
        st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=30,
    )
)
def test_basic_stats_percentiles_monotonic(pets):
    analyzer = _make_analyzer(pets)
    s = analyzer.basic_stats()

    pcts = [1, 5, 10, 90, 95, 99]
    values = [s[f"p{p}"] for p in pcts]
    assert values == sorted(values)


# ------------------------------------------------------------------ #
# risk_assessment
# ------------------------------------------------------------------ #
@given(
    st.lists(
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=30,
    )
)
def test_risk_assessment_total_and_exclusive(pets):
    analyzer = _make_analyzer(pets)
    risk_df = analyzer.risk_assessment()

    assert len(risk_df) == len(pets)
    assert set(risk_df["risk_level"]).issubset(
        {"Critical", "Serious", "Moderate", "Safe"}
    )


# ------------------------------------------------------------------ #
# risk_summary
# ------------------------------------------------------------------ #
@given(
    st.lists(
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=30,
    )
)
def test_risk_summary_counts_match_and_percentages_sum_to_100(pets):
    analyzer = _make_analyzer(pets)
    summary = analyzer.risk_summary()

    total = sum(summary[k]["count"] for k in ["critical", "serious", "moderate", "safe"])
    assert total == len(pets)

    pct_total = sum(
        summary[k]["percentage"] for k in ["critical", "serious", "moderate", "safe"]
    )
    assert abs(pct_total - 100.0) < 1e-6

    assert summary["conflict_rate"]["count"] == (
        summary["critical"]["count"] + summary["serious"]["count"]
    )


# ------------------------------------------------------------------ #
# by_conflict_type
# ------------------------------------------------------------------ #
@given(
    st.lists(
        st.tuples(
            st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
            st.sampled_from(["A", "B", "C"]),
        ),
        min_size=2,
        max_size=20,
    )
)
def test_by_conflict_type_sorted_by_conflict_rate(rows):
    pets = [p for p, _ in rows]
    types = [t for _, t in rows]
    analyzer = _make_analyzer(pets, types)
    out = analyzer.by_conflict_type()

    if not out.empty:
        rates = list(out["conflict_rate"])
        assert rates == sorted(rates, reverse=True)
        assert int(out["count"].sum()) == len(pets)
