"""Assemble the 15-row "gold-standard" evaluation table.

Pure functions; no I/O, no plotting, no file reads. Each metric family is
supplied as a mapping of well-known keys to numeric values, and the builder
produces a stable, ordered list of rows.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

TIER_DETECTION = "Detection"
TIER_TRACKING = "Tracking"
TIER_TRAJECTORY = "Trajectory"
TIER_SAFETY = "Safety"

ALL_TIERS: frozenset[str] = frozenset({TIER_DETECTION, TIER_TRACKING, TIER_TRAJECTORY, TIER_SAFETY})


@dataclass(frozen=True)
class GoldStandardRow:
    metric: str
    value: float
    unit: str
    tier: str


_SPEC: tuple[tuple[str, str, str, str, str], ...] = (
    ("precision", "detection", "precision", "ratio", TIER_DETECTION),
    ("recall", "detection", "recall", "ratio", TIER_DETECTION),
    ("f1", "detection", "f1", "ratio", TIER_DETECTION),
    ("mAP50", "detection", "map50", "ratio", TIER_DETECTION),
    ("mAP50:95", "detection", "map50_95", "ratio", TIER_DETECTION),
    ("AP75", "detection", "ap75", "ratio", TIER_DETECTION),
    ("HOTA", "tracking", "hota", "ratio", TIER_TRACKING),
    ("IDF1", "tracking", "idf1", "ratio", TIER_TRACKING),
    ("MOTA", "tracking", "mota", "ratio", TIER_TRACKING),
    ("position_RMSE", "trajectory", "position_rmse_m", "m", TIER_TRAJECTORY),
    ("velocity_MAE", "trajectory", "velocity_mae_mps", "m/s", TIER_TRAJECTORY),
    ("accel_MAE", "trajectory", "accel_mae_mps2", "m/s^2", TIER_TRAJECTORY),
    ("PET_MAE", "ssm", "pet_mae_s", "s", TIER_SAFETY),
    ("TTC_MAE", "ssm", "ttc_mae_s", "s", TIER_SAFETY),
    ("critical_conflict_recall", "ssm", "critical_conflict_recall", "ratio", TIER_SAFETY),
)

N_ROWS = len(_SPEC)


def _safe_float(x: object) -> float:
    try:
        f = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    if f != f or f in (float("inf"), float("-inf")):
        return 0.0
    return f


def build_table(
    detection: Mapping[str, float] | None = None,
    tracking: Mapping[str, float] | None = None,
    trajectory: Mapping[str, float] | None = None,
    ssm: Mapping[str, float] | None = None,
) -> list[GoldStandardRow]:
    families: dict[str, Mapping[str, float] | None] = {
        "detection": detection,
        "tracking": tracking,
        "trajectory": trajectory,
        "ssm": ssm,
    }
    rows: list[GoldStandardRow] = []
    for row_id, family, key, unit, tier in _SPEC:
        fam = families.get(family) or {}
        rows.append(
            GoldStandardRow(
                metric=row_id,
                value=_safe_float(fam.get(key, 0.0)),
                unit=unit,
                tier=tier,
            )
        )
    return rows


def to_dicts(rows: list[GoldStandardRow]) -> list[dict[str, object]]:
    return [asdict(r) for r in rows]


def to_markdown(rows: list[GoldStandardRow]) -> str:
    lines = [
        "| Metric | Value | Unit | Tier |",
        "| --- | ---: | --- | --- |",
    ]
    for r in rows:
        lines.append(f"| {r.metric} | {r.value:.4f} | {r.unit} | {r.tier} |")
    return "\n".join(lines)
