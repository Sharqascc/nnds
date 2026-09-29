"""Track stability diagnostics. No ground truth required.

Measures internal properties of tracks produced by the pipeline:
  - length distribution (fragmentation)
  - gap distribution (breaks)
  - trajectory smoothness (jerk / curvature spikes -> ID switch signal)
  - box-size stability (sudden w/h change -> ID switch or detection drift)
  - jump rate (center displacement between adjacent frames)

These are NOT accuracy metrics. They say whether tracks behave
consistently, which is a proxy for "is the tracker falling apart".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrackPointRow:
    frame: int
    cx: float
    cy: float
    w: float
    h: float


def _sorted_by_frame(pts: Sequence[TrackPointRow]) -> list[TrackPointRow]:
    return sorted(pts, key=lambda p: p.frame)


def length_histogram(tracks: Mapping[int, Sequence[TrackPointRow]]) -> dict[str, float]:
    if not tracks:
        return {
            "n_tracks": 0,
            "mean": 0.0,
            "median": 0.0,
            "max": 0.0,
            "min": 0.0,
            "n_short": 0,
            "frac_short": 0.0,
        }
    lengths = [len(v) for v in tracks.values()]
    n_short = sum(1 for L in lengths if L < 10)
    return {
        "n_tracks": len(lengths),
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "max": float(np.max(lengths)),
        "min": float(np.min(lengths)),
        "n_short": int(n_short),
        "frac_short": float(n_short / len(lengths)),
    }


def gap_metrics(
    tracks: Mapping[int, Sequence[TrackPointRow]], max_gap: int = 3
) -> dict[str, float]:
    gaps: list[int] = []
    tracks_with_gap = 0
    for pts in tracks.values():
        s = _sorted_by_frame(pts)
        if len(s) < 2:
            continue
        has_gap = False
        for i in range(1, len(s)):
            g = s[i].frame - s[i - 1].frame
            if g > 1:
                gaps.append(g)
                if g > max_gap:
                    has_gap = True
        if has_gap:
            tracks_with_gap += 1
    n_tracks = max(len(tracks), 1)
    return {
        "n_gaps": len(gaps),
        "mean_gap": float(np.mean(gaps)) if gaps else 0.0,
        "max_gap": int(np.max(gaps)) if gaps else 0,
        "tracks_with_large_gap": tracks_with_gap,
        "frac_tracks_with_large_gap": float(tracks_with_gap / n_tracks),
    }


def jump_metrics(
    tracks: Mapping[int, Sequence[TrackPointRow]], threshold_px: float = 30.0
) -> dict[str, float]:
    """Center displacement between adjacent frames, normalized per track."""
    all_jumps: list[float] = []
    n_large = 0
    for pts in tracks.values():
        s = _sorted_by_frame(pts)
        for i in range(1, len(s)):
            if s[i].frame - s[i - 1].frame != 1:
                continue
            d = float(np.hypot(s[i].cx - s[i - 1].cx, s[i].cy - s[i - 1].cy))
            all_jumps.append(d)
            if d > threshold_px:
                n_large += 1
    return {
        "n_jumps": len(all_jumps),
        "mean_jump_px": float(np.mean(all_jumps)) if all_jumps else 0.0,
        "p95_jump_px": float(np.percentile(all_jumps, 95)) if all_jumps else 0.0,
        "max_jump_px": float(np.max(all_jumps)) if all_jumps else 0.0,
        "n_large_jumps": n_large,
    }


def smoothness_metrics(tracks: Mapping[int, Sequence[TrackPointRow]]) -> dict[str, float]:
    """Per-track second-difference (acceleration) magnitude, in px.

    A smooth track has low acceleration magnitude. A sudden ID switch shows
    up as a large spike in the second difference.
    """
    accs: list[float] = []
    per_track_p95: list[float] = []
    for pts in tracks.values():
        s = _sorted_by_frame(pts)
        if len(s) < 3:
            continue
        # Only consider consecutive frames
        seq: list[TrackPointRow] = [s[0]]
        for i in range(1, len(s)):
            if s[i].frame - s[i - 1].frame == 1:
                seq.append(s[i])
            else:
                if len(seq) >= 3:
                    a = _second_diff(seq)
                    accs.extend(a)
                    per_track_p95.append(float(np.percentile(a, 95)))
                seq = [s[i]]
        if len(seq) >= 3:
            a = _second_diff(seq)
            accs.extend(a)
            per_track_p95.append(float(np.percentile(a, 95)))
    return {
        "n_accels": len(accs),
        "mean_accel_px": float(np.mean(accs)) if accs else 0.0,
        "p95_accel_px": float(np.percentile(accs, 95)) if accs else 0.0,
        "max_accel_px": float(np.max(accs)) if accs else 0.0,
        "mean_per_track_p95": float(np.mean(per_track_p95)) if per_track_p95 else 0.0,
    }


def _second_diff(seq: Sequence[TrackPointRow]) -> list[float]:
    xs = np.array([p.cx for p in seq], dtype=float)
    ys = np.array([p.cy for p in seq], dtype=float)
    ax = xs[2:] - 2 * xs[1:-1] + xs[:-2]
    ay = ys[2:] - 2 * ys[1:-1] + ys[:-2]
    return [float(v) for v in np.hypot(ax, ay)]


def box_size_metrics(
    tracks: Mapping[int, Sequence[TrackPointRow]], threshold_frac: float = 0.5
) -> dict[str, float]:
    """Per-frame relative change of box width and height within a track."""
    changes: list[float] = []
    n_large = 0
    for pts in tracks.values():
        s = _sorted_by_frame(pts)
        for i in range(1, len(s)):
            if s[i].frame - s[i - 1].frame != 1:
                continue
            w0 = max(s[i - 1].w, 1.0)
            h0 = max(s[i - 1].h, 1.0)
            dw = abs(s[i].w - w0) / w0
            dh = abs(s[i].h - h0) / h0
            v = max(dw, dh)
            changes.append(v)
            if v > threshold_frac:
                n_large += 1
    return {
        "n_box_changes": len(changes),
        "mean_rel_change": float(np.mean(changes)) if changes else 0.0,
        "p95_rel_change": float(np.percentile(changes, 95)) if changes else 0.0,
        "n_large_changes": n_large,
    }


def full_report(tracks: Mapping[int, Sequence[TrackPointRow]]) -> dict[str, dict[str, float]]:
    return {
        "length": length_histogram(tracks),
        "gaps": gap_metrics(tracks),
        "jumps": jump_metrics(tracks),
        "smoothness": smoothness_metrics(tracks),
        "box_size": box_size_metrics(tracks),
    }


def tracks_from_rows(rows: Sequence[dict]) -> dict[int, list[TrackPointRow]]:
    """Build {track_id -> [TrackPointRow]} from detection rows.

    Detection rows must contain: frame, track_id, cx, cy, w, h
    """
    out: dict[int, list[TrackPointRow]] = {}
    for r in rows:
        tid = int(r["track_id"])
        out.setdefault(tid, []).append(
            TrackPointRow(
                frame=int(r["frame"]),
                cx=float(r["cx"]),
                cy=float(r["cy"]),
                w=float(r["w"]),
                h=float(r["h"]),
            )
        )
    return out
