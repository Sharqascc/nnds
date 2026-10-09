"""Integration: gate config -> gate counter -> crossing events.

Runs in PR CI (no heavy assets).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]


def test_gate_config_parses_for_root():
    p = REPO / "configs" / "gate_config.yaml"
    assert p.exists()
    cfg = yaml.safe_load(p.read_text())
    assert isinstance(cfg, dict)


def test_site_gate_configs_parse():
    for site in ("giti", "mrc"):
        p = REPO / "configs" / "sites" / site / "gate_config.yaml"
        if not p.exists():
            pytest.skip(f"missing {p}")
        cfg = yaml.safe_load(p.read_text())
        assert isinstance(cfg, dict)


def test_gate_counter_module_public_api():
    from src.analysis import gate_counter

    names = [n for n in dir(gate_counter) if not n.startswith("_")]
    assert any("count" in n.lower() or "gate" in n.lower() for n in names), names


def test_gate_counter_module_has_gate_dataclass_or_class():
    from src.analysis import gate_counter

    # At least one class or dataclass
    classes = [
        n
        for n in dir(gate_counter)
        if not n.startswith("_") and isinstance(getattr(gate_counter, n, None), type)
    ]
    assert classes, "no classes found in gate_counter"
