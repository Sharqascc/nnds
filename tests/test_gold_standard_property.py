from hypothesis import given, settings
from hypothesis import strategies as st

from src.analysis.gold_standard import (
    ALL_TIERS,
    N_ROWS,
    build_table,
    to_dicts,
    to_markdown,
)

_FAMILY_KEYS = {
    "detection": ["precision", "recall", "f1", "map50", "map50_95", "ap75"],
    "tracking": ["hota", "idf1", "mota"],
    "trajectory": ["position_rmse_m", "velocity_mae_mps", "accel_mae_mps2"],
    "ssm": ["pet_mae_s", "ttc_mae_s", "critical_conflict_recall"],
}


def _family_st(keys):
    return st.one_of(
        st.none(),
        st.dictionaries(
            keys=st.sampled_from(keys),
            values=st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False),
            max_size=len(keys),
        ),
    )


families_st = st.tuples(
    _family_st(_FAMILY_KEYS["detection"]),
    _family_st(_FAMILY_KEYS["tracking"]),
    _family_st(_FAMILY_KEYS["trajectory"]),
    _family_st(_FAMILY_KEYS["ssm"]),
)


@given(families_st)
@settings(max_examples=50)
def test_always_fifteen_rows(families):
    rows = build_table(*families)
    assert len(rows) == N_ROWS


@given(families_st)
@settings(max_examples=50)
def test_all_tiers_present(families):
    rows = build_table(*families)
    assert {r.tier for r in rows} == ALL_TIERS


@given(families_st)
@settings(max_examples=50)
def test_values_are_finite_floats(families):
    rows = build_table(*families)
    for r in rows:
        assert isinstance(r.value, float)
        assert r.value == r.value
        assert r.value not in (float("inf"), float("-inf"))


@given(families_st)
@settings(max_examples=50)
def test_markdown_line_count(families):
    md = to_markdown(build_table(*families))
    assert len(md.splitlines()) == 2 + N_ROWS


@given(families_st)
@settings(max_examples=50)
def test_dicts_have_expected_fields(families):
    d = to_dicts(build_table(*families))
    for row in d:
        assert set(row.keys()) == {"metric", "value", "unit", "tier"}


@given(
    st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
    st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
    st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50)
def test_tracking_values_preserved(hota, idf1, mota):
    rows = {
        r.metric: r.value for r in build_table(tracking={"hota": hota, "idf1": idf1, "mota": mota})
    }
    assert abs(rows["HOTA"] - hota) < 1e-12
    assert abs(rows["IDF1"] - idf1) < 1e-12
    assert abs(rows["MOTA"] - mota) < 1e-12
