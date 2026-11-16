"""Property tests for src/analysis/_stages.py.

Complements tests/test_stages_progress.py (unit tests, one input at a time)
by exercising the invariants across a wide input space with Hypothesis.

Property focus:
  * No crash on any bounded input (integer / printable text).
  * Global-state mutations match the documented contract.
  * `progress()` no-op conditions (quiet mode, current<=0) hold universally.
  * `CANONICAL_STAGES` remains well-formed under any future edit.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.analysis import _stages as stages


STAGE_NUM = st.integers(min_value=0, max_value=20)
TOTAL = st.integers(min_value=1, max_value=20)
EVERY = st.integers(min_value=1, max_value=20)
CURRENT = st.integers(min_value=1, max_value=500)
LABEL = st.text(
    min_size=0,
    max_size=40,
    alphabet=st.characters(min_codepoint=32, max_codepoint=126),
)
HEADER = LABEL


@pytest.fixture(autouse=True)
def _reset_stage_globals():
    """Reset module-level state before/after each test (avoids order dependence)."""
    stages._TOTAL = None
    stages._START = None
    stages._QUIET = False
    stages._CURRENT_STAGE = 0
    yield
    stages._TOTAL = None
    stages._START = None
    stages._QUIET = False
    stages._CURRENT_STAGE = 0


@pytest.mark.property
@given(total=TOTAL, header=HEADER)
@settings(max_examples=20, deadline=None)
def test_begin_never_raises(total, header):
    with redirect_stdout(io.StringIO()):
        stages.begin(total, header=header)


@pytest.mark.property
@given(total=TOTAL, header=HEADER)
@settings(max_examples=20, deadline=None)
def test_begin_sets_total_and_resets_stage(total, header):
    with redirect_stdout(io.StringIO()):
        stages.begin(total, header=header)
    assert stages._TOTAL == total
    assert stages._CURRENT_STAGE == 0


@pytest.mark.property
@given(n=STAGE_NUM, name=LABEL, detail=LABEL)
@settings(max_examples=20, deadline=None)
def test_stage_never_raises(n, name, detail):
    with redirect_stdout(io.StringIO()):
        stages.stage(n, name, detail)


@pytest.mark.property
@given(n=STAGE_NUM, name=LABEL)
@settings(max_examples=20, deadline=None)
def test_stage_sets_current_stage(n, name):
    with redirect_stdout(io.StringIO()):
        stages.stage(n, name)
    assert stages._CURRENT_STAGE == n


@pytest.mark.property
@given(
    current=CURRENT,
    total=st.integers(min_value=0, max_value=100),
    every=EVERY,
    label=LABEL,
)
@settings(max_examples=20, deadline=None)
def test_progress_never_raises(current, total, every, label):
    with redirect_stdout(io.StringIO()):
        stages.begin(3, quiet=False)
        stages.progress(current, total, label, every=every)


@pytest.mark.property
@given(
    current=CURRENT,
    total=st.integers(min_value=0, max_value=100),
    every=EVERY,
    label=LABEL,
)
@settings(max_examples=20, deadline=None)
def test_progress_quiet_suppresses_stage_output(current, total, every, label):
    buf = io.StringIO()
    with redirect_stdout(buf):
        stages.begin(3, quiet=True)
        stages.progress(current, total, label, every=every)
    out = buf.getvalue()
    assert "[stage " not in out
    assert "[progress]" not in out


@pytest.mark.property
@given(current=st.integers(min_value=-1000, max_value=0), label=LABEL)
@settings(max_examples=20, deadline=None)
def test_progress_non_positive_current_is_noop(current, label):
    buf = io.StringIO()
    with redirect_stdout(buf):
        stages.begin(3, quiet=False)
        stages.progress(current, 100, label, every=1)
    out = buf.getvalue()
    assert "[stage " not in out
    assert "[progress]" not in out


@pytest.mark.property
@given(detail=LABEL)
@settings(max_examples=20, deadline=None)
def test_done_never_raises(detail):
    with redirect_stdout(io.StringIO()):
        stages.done(detail)


@pytest.mark.property
@given(data=st.data())
@settings(max_examples=20, deadline=None)
def test_canonical_stages_are_unique_pairwise(data):
    """Any two distinct valid indices yield distinct stage names."""
    n = len(stages.CANONICAL_STAGES)
    if n < 2:
        return
    i = data.draw(st.integers(min_value=1, max_value=n))
    j = data.draw(st.integers(min_value=1, max_value=n).filter(lambda x: x != i))
    assert stages.CANONICAL_STAGES[i - 1] != stages.CANONICAL_STAGES[j - 1]


@pytest.mark.property
@given(data=st.data())
@settings(max_examples=20, deadline=None)
def test_canonical_stages_entries_are_non_empty_strings(data):
    n = len(stages.CANONICAL_STAGES)
    if n < 1:
        return
    idx = data.draw(st.integers(min_value=1, max_value=n))
    entry = stages.CANONICAL_STAGES[idx - 1]
    assert isinstance(entry, str)
    assert len(entry.strip()) > 0
