from hypothesis import given, settings
from hypothesis import strategies as st

from src.analysis.gt_validation import (
    has_errors,
    validate_detection_rows,
    validate_ssm_rows,
    validate_tracking_rows,
)


def _det(frame, x1, y1, w, h):
    return {
        "frame": frame,
        "x1": x1,
        "y1": y1,
        "x2": x1 + w,
        "y2": y1 + h,
        "class_name": "car",
    }


@given(
    st.lists(
        st.tuples(
            st.integers(0, 20),
            st.floats(0, 500, allow_nan=False, allow_infinity=False),
            st.floats(0, 500, allow_nan=False, allow_infinity=False),
            st.floats(1, 200, allow_nan=False, allow_infinity=False),
            st.floats(1, 200, allow_nan=False, allow_infinity=False),
        ),
        min_size=1,
        max_size=15,
    ).map(sorted)
)
@settings(max_examples=50)
def test_well_formed_detection_has_no_errors(rows):
    dets = [_det(*r) for r in rows]
    assert not has_errors(validate_detection_rows(dets))


@given(
    st.lists(
        st.tuples(
            st.integers(0, 20),
            st.integers(1, 5),
            st.floats(0, 500, allow_nan=False, allow_infinity=False),
            st.floats(0, 500, allow_nan=False, allow_infinity=False),
            st.floats(1, 200, allow_nan=False, allow_infinity=False),
            st.floats(1, 200, allow_nan=False, allow_infinity=False),
        ),
        min_size=2,
        max_size=15,
    )
)
@settings(max_examples=50)
def test_well_formed_tracking_has_no_errors(rows):
    trk = [
        {"frame": r[0], "track_id": r[1], "x": r[2], "y": r[3], "w": r[4], "h": r[5]} for r in rows
    ]
    assert not has_errors(validate_tracking_rows(trk))


@given(
    st.lists(
        st.tuples(
            st.integers(1, 10),
            st.integers(1, 10),
            st.floats(0.01, 10.0, allow_nan=False, allow_infinity=False),
        ),
        min_size=1,
        max_size=10,
    )
)
@settings(max_examples=50)
def test_well_formed_ssm_has_no_errors(rows):
    ssm = [{"track_a": a, "track_b": b, "pet": pet} for a, b, pet in rows if a != b]
    if not ssm:
        return
    assert not has_errors(validate_ssm_rows(ssm))


@given(st.integers(0, 30))
@settings(max_examples=30)
def test_missing_column_always_errors(frame):
    assert has_errors(validate_detection_rows([{"frame": frame}]))
    assert has_errors(validate_tracking_rows([{"frame": frame}]))
    assert has_errors(validate_ssm_rows([{"track_a": 1}]))
