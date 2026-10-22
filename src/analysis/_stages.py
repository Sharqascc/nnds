"""Minimal numbered-stage output for pipeline orchestration.

Used by the grid-PET runners so a long pipeline prints a stable, parseable
progress trace:

    ============================================================
    🚦 NNDS - GITI_traffic_video.mp4
    ============================================================
    [1/5] Setup              backend=pytorch device=cpu
    [2/5] Detect + Track     max_frames=300
    [3/5] Index tracks
    [4/5] Compute PET        threshold=3.0s tracks=42
    [5/5] Write outputs      outputs/petevents_bev.csv
    ============================================================
    ✅ Done in 94.2s  —  18 events -> outputs/petevents_bev.csv
    ============================================================

Nothing here changes pipeline behavior - the functions only print. The tests
in tests/test_stages.py assert the format is stable.
"""

from __future__ import annotations

import time
from typing import Optional

_TOTAL: int | None = None
_START: float | None = None


def begin(total: int, header: str = "NNDS Pipeline") -> None:
    """Print a header and record the total stage count and start time."""
    global _TOTAL, _START
    _TOTAL = total
    _START = time.monotonic()
    line = "=" * 60
    print(line)
    print(f"🚦 {header}")
    print(line, flush=True)


def stage(n: int, name: str, detail: str = "") -> None:
    """Print one numbered stage line: `[n/total] name    detail`."""
    total = _TOTAL if _TOTAL is not None else "?"
    label = f"[{n}/{total}]"
    if detail:
        print(f"{label} {name:<18} {detail}", flush=True)
    else:
        print(f"{label} {name}", flush=True)


def done(detail: str = "") -> None:
    """Print the closing banner with elapsed wall time."""
    elapsed = time.monotonic() - _START if _START is not None else 0.0
    line = "=" * 60
    print(line)
    if detail:
        print(f"✅ Done in {elapsed:.1f}s  —  {detail}")
    else:
        print(f"✅ Done in {elapsed:.1f}s")
    print(line, flush=True)
