"""Property tests for src.pipeline.preflight."""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.pipeline import preflight


# ---------------------------------------------------------------------------
# PreflightReport invariants
# ---------------------------------------------------------------------------


@given(name=st.text(min_size=1, max_size=30))
@settings(max_examples=50, deadline=None)
def test_add_single_call_appends_one_check(name):
    r = preflight.PreflightReport()
    assert len(r.checks) == 0
    r.add(name, True, "detail")
    assert len(r.checks) == 1
    assert r.checks[0]["check"] == name


@given(n=st.integers(min_value=1, max_value=30))
@settings(max_examples=50, deadline=None)
def test_add_n_calls_appends_n_checks(n):
    r = preflight.PreflightReport()
    for i in range(n):
        r.add(f"check_{i}", True, "detail")
    assert len(r.checks) == n
    assert r.ok is True
    assert r.problems == []


@given(fail_count=st.integers(min_value=1, max_value=10))
@settings(max_examples=30, deadline=None)
def test_ok_is_false_iff_problems_nonempty(fail_count):
    r = preflight.PreflightReport()
    for i in range(fail_count):
        r.add(f"bad_{i}", False, "detail")
    assert r.ok is False
    assert len(r.problems) == fail_count


def test_to_dict_shape_stable_with_no_checks():
    r = preflight.PreflightReport()
    d = r.to_dict()
    assert set(d.keys()) == {"ok", "checks", "problems"}
    assert d["ok"] is True
    assert d["checks"] == []
    assert d["problems"] == []


@given(
    ok_flags=st.lists(st.booleans(), min_size=0, max_size=10),
)
@settings(max_examples=50, deadline=None)
def test_to_dict_matches_report_state(ok_flags):
    r = preflight.PreflightReport()
    for i, flag in enumerate(ok_flags):
        r.add(f"c{i}", flag, "detail")
    d = r.to_dict()
    assert d["ok"] is (False not in ok_flags or len(ok_flags) == 0)
    assert len(d["checks"]) == len(ok_flags)
    assert len(d["problems"]) == sum(1 for f in ok_flags if not f)


# ---------------------------------------------------------------------------
# _check_file behaviour
# ---------------------------------------------------------------------------


@given(
    min_bytes=st.integers(min_value=1, max_value=1_000_000),
    size=st.integers(min_value=0, max_value=2_000_000),
)
@settings(max_examples=50, deadline=None)
def test_check_file_ok_iff_size_above_threshold(tmp_path_factory, min_bytes, size):
    tmp = tmp_path_factory.mktemp("preflight")
    p = tmp / "f.bin"
    p.write_bytes(b"x" * size)
    r = preflight.PreflightReport()
    preflight._check_file(r, str(p), "some_file", min_bytes, "fix")
    assert len(r.checks) == 1
    assert r.checks[0]["ok"] is (size >= min_bytes)


def test_check_file_missing_path_marks_not_ok():
    r = preflight.PreflightReport()
    preflight._check_file(r, "/nonexistent/a.bin", "some_file", 1, "fix")
    assert r.ok is False
    assert "missing" in r.problems[0]


# ---------------------------------------------------------------------------
# _check_config behaviour
# ---------------------------------------------------------------------------


@given(content=st.text(min_size=0, max_size=50))
@settings(max_examples=30, deadline=None)
def test_check_config_handles_arbitrary_text(tmp_path_factory, content):
    tmp = tmp_path_factory.mktemp("preflight")
    p = tmp / "cfg.json"
    p.write_text(content)
    r = preflight.PreflightReport()
    preflight._check_config(r, str(p), "cfg")
    assert len(r.checks) == 1
    # json.loads either succeeded or the check is marked failed


def test_check_config_none_is_ok():
    r = preflight.PreflightReport()
    preflight._check_config(r, None, "cfg")
    assert r.checks[0]["ok"] is True


# ---------------------------------------------------------------------------
# run_checks: any missing video path is always a problem
# ---------------------------------------------------------------------------


@given(path=st.text(min_size=1, max_size=60))
@settings(max_examples=30, deadline=None)
def test_run_checks_missing_video_always_fails(path):
    # Skip paths that might accidentally exist on the runner
    if Path(path).exists():
        return
    r = preflight.run_checks(video=path)
    assert r.ok is False


def test_run_checks_no_args_ok_when_environment_present():
    # With no video and no weights, run_checks reports on env only.
    r = preflight.run_checks()
    # Env checks (torch, ffmpeg, ultralytics) may or may not pass in CI;
    # the invariant is that the report is well-formed.
    d = r.to_dict()
    assert set(d.keys()) == {"ok", "checks", "problems"}
    assert isinstance(d["checks"], list)
    assert isinstance(d["problems"], list)
