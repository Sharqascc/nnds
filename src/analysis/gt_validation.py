"""Ground-truth CSV validation. Pure functions; no I/O."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

REQUIRED_DETECTION_COLS: frozenset[str] = frozenset({"frame", "x1", "y1", "x2", "y2", "class_name"})
REQUIRED_TRACKING_COLS: frozenset[str] = frozenset({"frame", "track_id", "x", "y", "w", "h"})
REQUIRED_TRAJECTORY_COLS: frozenset[str] = frozenset({"frame", "track_id", "x", "y"})
REQUIRED_SSM_COLS: frozenset[str] = frozenset({"track_a", "track_b", "pet"})

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    kind: str
    message: str
    row: int | None = None


def _is_finite_number(x: object) -> bool:
    try:
        f = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return f == f and f not in (float("inf"), float("-inf"))


def _to_int(x: object, default: int = -1) -> int:
    try:
        return int(x)  # type: ignore[call-overload, arg-type]
    except (TypeError, ValueError):
        return default


def _to_float(x: object, default: float = 0.0) -> float:
    try:
        return float(x)  # type: ignore[call-overload, arg-type]
    except (TypeError, ValueError):
        return default


def _check_columns(row: dict, required: frozenset[str]) -> list[ValidationIssue]:
    missing = required - set(row.keys())
    if missing:
        return [
            ValidationIssue(
                SEVERITY_ERROR,
                "missing_columns",
                f"missing required columns: {sorted(missing)}",
            )
        ]
    return []


def _check_frame_monotonic(frames: Sequence[int]) -> list[ValidationIssue]:
    if list(frames) != sorted(frames):
        return [
            ValidationIssue(
                SEVERITY_WARNING,
                "unsorted_frames",
                "frames are not monotonically non-decreasing",
            )
        ]
    return []


def _check_coverage_gaps(frames: Sequence[int], max_gap: int = 10) -> list[ValidationIssue]:
    if len(frames) < 2:
        return []
    issues: list[ValidationIssue] = []
    for i in range(1, len(frames)):
        gap = frames[i] - frames[i - 1]
        if gap > max_gap:
            issues.append(
                ValidationIssue(
                    SEVERITY_WARNING,
                    "large_gap",
                    f"frame gap {gap} between {frames[i - 1]} and {frames[i]}",
                )
            )
    return issues


def validate_detection_rows(rows: Sequence[dict]) -> list[ValidationIssue]:
    if not rows:
        return [ValidationIssue(SEVERITY_WARNING, "empty", "no rows in detection GT")]
    issues = _check_columns(rows[0], REQUIRED_DETECTION_COLS)
    if issues:
        return issues

    frames: list[int] = []
    seen: set[tuple] = set()
    for i, r in enumerate(rows):
        for k in ("frame", "x1", "y1", "x2", "y2"):
            if not _is_finite_number(r.get(k)):
                issues.append(ValidationIssue(SEVERITY_ERROR, "non_finite", f"{k} not finite", i))
        x1 = float(r.get("x1", 0.0))
        y1 = float(r.get("y1", 0.0))
        x2 = float(r.get("x2", 0.0))
        y2 = float(r.get("y2", 0.0))
        if x2 <= x1 or y2 <= y1:
            issues.append(
                ValidationIssue(
                    SEVERITY_ERROR,
                    "invalid_box",
                    f"degenerate box: ({x1},{y1},{x2},{y2})",
                    i,
                )
            )
        frame = int(r.get("frame", 0))
        frames.append(frame)
        key = (frame, x1, y1, x2, y2, str(r.get("class_name", "")))
        if key in seen:
            issues.append(ValidationIssue(SEVERITY_WARNING, "duplicate", "duplicate row", i))
        seen.add(key)
    issues.extend(_check_frame_monotonic(frames))
    issues.extend(_check_coverage_gaps(frames))
    return issues


def validate_tracking_rows(rows: Sequence[dict]) -> list[ValidationIssue]:
    if not rows:
        return [ValidationIssue(SEVERITY_WARNING, "empty", "no rows in tracking GT")]
    issues = _check_columns(rows[0], REQUIRED_TRACKING_COLS)
    if issues:
        return issues

    frames: list[int] = []
    by_track: dict[int, list[int]] = {}
    for i, r in enumerate(rows):
        for k in ("frame", "track_id", "x", "y", "w", "h"):
            if not _is_finite_number(r.get(k)):
                issues.append(ValidationIssue(SEVERITY_ERROR, "non_finite", f"{k} not finite", i))
        w = float(r.get("w", 0.0))
        h = float(r.get("h", 0.0))
        if w <= 0 or h <= 0:
            issues.append(
                ValidationIssue(
                    SEVERITY_ERROR,
                    "invalid_box",
                    f"non-positive w/h: ({w},{h})",
                    i,
                )
            )
        frame = int(r.get("frame", 0))
        tid = int(r.get("track_id", -1))
        frames.append(frame)
        by_track.setdefault(tid, []).append(frame)

    for tid, fs in by_track.items():
        if len(fs) < 2:
            issues.append(
                ValidationIssue(
                    SEVERITY_WARNING,
                    "short_track",
                    f"track_id {tid} has only {len(fs)} frame(s)",
                )
            )

    issues.extend(_check_frame_monotonic(frames))
    issues.extend(_check_coverage_gaps(sorted(set(frames))))
    return issues


def validate_trajectory_rows(rows: Sequence[dict]) -> list[ValidationIssue]:
    if not rows:
        return [ValidationIssue(SEVERITY_WARNING, "empty", "no rows in trajectory GT")]
    issues = _check_columns(rows[0], REQUIRED_TRAJECTORY_COLS)
    if issues:
        return issues
    frames: list[int] = []
    for i, r in enumerate(rows):
        for k in ("frame", "track_id", "x", "y"):
            if not _is_finite_number(r.get(k)):
                issues.append(ValidationIssue(SEVERITY_ERROR, "non_finite", f"{k} not finite", i))
        frames.append(int(r.get("frame", 0)))
    issues.extend(_check_frame_monotonic(frames))
    return issues


def validate_ssm_rows(
    rows: Sequence[dict],
    known_track_ids: set[int] | None = None,
) -> list[ValidationIssue]:
    if not rows:
        return [ValidationIssue(SEVERITY_WARNING, "empty", "no rows in SSM GT")]
    issues = _check_columns(rows[0], REQUIRED_SSM_COLS)
    if issues:
        return issues

    for i, r in enumerate(rows):
        a = r.get("track_a")
        b = r.get("track_b")
        if not _is_finite_number(a) or not _is_finite_number(b):
            issues.append(ValidationIssue(SEVERITY_ERROR, "non_finite", "track_a/b not finite", i))
            continue
        if _to_int(a) == _to_int(b):
            issues.append(
                ValidationIssue(
                    SEVERITY_ERROR,
                    "same_track",
                    f"track_a == track_b == {_to_int(a)}",
                    i,
                )
            )
        if known_track_ids is not None:
            for k in ("track_a", "track_b"):
                tid = _to_int(r.get(k, -1))
                if tid not in known_track_ids:
                    issues.append(
                        ValidationIssue(
                            SEVERITY_WARNING,
                            "unknown_track",
                            f"{k}={tid} not present in tracking GT",
                            i,
                        )
                    )
        pet = r.get("pet")
        if not _is_finite_number(pet):
            issues.append(ValidationIssue(SEVERITY_ERROR, "non_finite", "pet not finite", i))
        elif _to_float(pet) <= 0:
            issues.append(
                ValidationIssue(
                    SEVERITY_ERROR,
                    "non_positive_pet",
                    f"pet must be > 0, got {pet}",
                    i,
                )
            )
        if "ttc" in r and r["ttc"] is not None:
            if not _is_finite_number(r["ttc"]):
                issues.append(ValidationIssue(SEVERITY_ERROR, "non_finite", "ttc not finite", i))
            elif _to_float(r["ttc"]) <= 0:
                issues.append(
                    ValidationIssue(
                        SEVERITY_ERROR,
                        "non_positive_ttc",
                        f"ttc must be > 0, got {r['ttc']}",
                        i,
                    )
                )
    return issues


def has_errors(issues: Sequence[ValidationIssue]) -> bool:
    return any(i.severity == SEVERITY_ERROR for i in issues)


def summarise(issues: Sequence[ValidationIssue]) -> dict[str, int]:
    counts = {SEVERITY_ERROR: 0, SEVERITY_WARNING: 0}
    for i in issues:
        counts[i.severity] = counts.get(i.severity, 0) + 1
    return counts
