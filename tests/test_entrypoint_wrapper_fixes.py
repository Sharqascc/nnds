"""Tests for entry-point wrapper review fixes (PR #49 findings)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


# ---------- helpers ----------


def _strip_line_comments(text: str, marker: str = "#") -> str:
    """Drop lines whose first non-space char is the comment marker.

    We check code, not prose: a comment explaining "we used to write to
    <bad-pattern>" must not trip a guard that looks for <bad-pattern>.
    """
    return "\n".join("" if line.lstrip().startswith(marker) else line for line in text.splitlines())


# ---------- run_pipeline._get_flag_value ----------


def test_get_flag_value_space_form():
    from scripts.run_pipeline import _get_flag_value

    assert _get_flag_value(["--foo", "bar"], "--foo") == "bar"


def test_get_flag_value_equals_form():
    from scripts.run_pipeline import _get_flag_value

    assert _get_flag_value(["--foo=bar"], "--foo") == "bar"


def test_get_flag_value_missing():
    from scripts.run_pipeline import _get_flag_value

    assert _get_flag_value(["--other", "x"], "--foo") is None


def test_get_flag_value_at_end_returns_none():
    from scripts.run_pipeline import _get_flag_value

    assert _get_flag_value(["--foo"], "--foo") is None


def test_get_flag_value_equals_empty_is_distinguishable():
    from scripts.run_pipeline import _get_flag_value

    assert _get_flag_value(["--foo="], "--foo") == ""


# ---------- reproduce_pipeline.sh (code lines only) ----------


def test_reproduce_script_uses_max_frames_in_filenames():
    raw = (REPO / "scripts" / "reproduce_pipeline.sh").read_text()
    code = _strip_line_comments(raw)
    # Output filenames must be built from a frame-count variable.
    assert "_full_${MAX_FRAMES}_parallel.csv" in code or "_full_{max_frames}_parallel.csv" in code
    # And the old hardcoded '300_parallel' literal must not appear in code.
    assert "300_parallel" not in code, (
        "reproduce_pipeline.sh code still uses a hardcoded 300 in a filename"
    )


def test_reproduce_script_respects_video_dir():
    raw = (REPO / "scripts" / "reproduce_pipeline.sh").read_text()
    code = _strip_line_comments(raw)
    assert "os.environ['GITI_VIDEO']" in code or 'os.environ["GITI_VIDEO"]' in code, (
        "reproduce_pipeline.sh does not consume GITI_VIDEO; NNDS_VIDEO_DIR "
        "would change preflight paths but not the paths the jobs use"
    )
    assert "os.environ['MRC_VIDEO']" in code or 'os.environ["MRC_VIDEO"]' in code


def test_reproduce_script_completion_respects_giti_only():
    raw = (REPO / "scripts" / "reproduce_pipeline.sh").read_text()
    code = _strip_line_comments(raw)
    lines = code.splitlines()
    # Find the completion block: for out in ...
    # Any MRC reference in the completion array must be inside a GITI_ONLY guard.
    in_completion = False
    for i, line in enumerate(lines):
        if 'echo "✅ Reproduction complete."' in line:
            in_completion = True
            continue
        if in_completion and "mrc" in line.lower() and "parallel.csv" in line:
            guard_present = any("GITI_ONLY" in lines[k] for k in range(max(0, i - 8), i))
            assert guard_present, (
                f"reproduce_pipeline.sh line {i + 1} lists MRC in completion "
                f"without a GITI_ONLY guard nearby"
            )
            break


# ---------- ci.yml (code lines only) ----------


def test_ci_pylint_targets_existing_directory():
    raw = (REPO / ".github" / "workflows" / "ci.yml").read_text()
    code = _strip_line_comments(raw)
    assert "pylint src/paper" not in code, (
        "ci.yml runs `pylint src/paper` but that directory does not exist"
    )
    assert "pylint paper/analysis" in code


def test_ci_coverage_combine_uses_append():
    raw = (REPO / ".github" / "workflows" / "ci.yml").read_text()
    code = _strip_line_comments(raw)
    # Any `coverage combine` in code must be followed on the same line by --append.
    for match in re.finditer(r"coverage combine([^\n]*)", code):
        rest = match.group(1)
        assert "--append" in rest, (
            f"`coverage combine{rest}` is not --append; the second aggregate overwrites the first"
        )


# ---------- cli-end-to-end.yml ----------


def test_cli_e2e_uses_named_gate_config():
    raw = (REPO / ".github" / "workflows" / "cli-end-to-end.yml").read_text()
    code = _strip_line_comments(raw)
    assert "configs/gate_config.yaml" not in code, (
        "cli-end-to-end.yml references configs/gate_config.yaml whose gates "
        "have no name field; gate annotations get dropped. Use "
        "configs/sites/giti/gate_config.yaml."
    )
    assert "configs/sites/giti/gate_config.yaml" in code


def test_cli_e2e_paths_include_detector():
    raw = (REPO / ".github" / "workflows" / "cli-end-to-end.yml").read_text()
    code = _strip_line_comments(raw)
    assert "src/analysis/grid_trajectory/uvh_coco_fused_grid_pet.py" in code
