"""Minimal numbered-stage output for pipeline orchestration.

Used by the grid-PET runners so a long pipeline prints a stable, parseable
progress trace. Functions print only; nothing here changes pipeline behavior.

Two levels of output:

  * `stage(N, name, ...)` — one line per phase (structural).
  * `progress(current, total, label, ...)` — periodic lines inside long loops
    (every `every` items, or at `current == total`).

`progress()` is a no-op when the pipeline was started with `quiet=True`
(`--no-progress` on the CLI). Stage lines still print, because they describe
structure, not granular progress.
"""

from __future__ import annotations

import time

_TOTAL: int | None = None
_START: float | None = None
_QUIET: bool = False
_CURRENT_STAGE: int = 0


def begin(total: int, header: str = "NNDS Pipeline", *, quiet: bool = False) -> None:
    """Print a header, record the total stage count, reset the stage counter."""
    global _TOTAL, _START, _QUIET, _CURRENT_STAGE
    _TOTAL = total
    _START = time.monotonic()
    _QUIET = quiet
    _CURRENT_STAGE = 0
    line = "=" * 60
    print(line)
    print(f"🚦 {header}")
    print(line, flush=True)


def stage(n: int, name: str, detail: str = "") -> None:
    """Print one numbered stage line: `[n/total] name    detail`."""
    global _CURRENT_STAGE
    _CURRENT_STAGE = n
    total = _TOTAL if _TOTAL is not None else "?"
    label = f"[{n}/{total}]"
    if detail:
        print(f"{label} {name:<18} {detail}", flush=True)
    else:
        print(f"{label} {name}", flush=True)


def progress(
    current: int,
    total: int,
    label: str,
    every: int = 100,
    *,
    stage_num: int | None = None,
) -> None:
    """Print `[stage k/N] label  current/total` every `every` items.

    No-op when `begin(quiet=True)` was used. `total <= 0` means "unknown";
    in that case the denominator prints as `?` and only the every-N trigger
    fires. `stage_num=0` forces the `[progress]` prefix (used for
    post-pipeline steps that are not one of the numbered stages).
    """
    if _QUIET:
        return
    if every <= 0:
        every = 100
    if current <= 0:
        return

    if total > 0:
        trigger = (current % every == 0) or (current == total)
        denom = str(total)
    else:
        trigger = current % every == 0
        denom = "?"

    if not trigger:
        return

    if stage_num is None:
        stage_num = _CURRENT_STAGE
    total_stages = _TOTAL if _TOTAL is not None else "?"
    prefix = f"[stage {stage_num}/{total_stages}]" if stage_num > 0 else "[progress]"
    print(f"{prefix} {label:<15} {current}/{denom}", flush=True)


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
