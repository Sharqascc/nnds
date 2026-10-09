"""Integration: config files parse and cross-reference each other.

No heavy assets. Runs in PR CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def test_root_bev_config_exists_and_parses():
    cfg = _load_json(CONFIGS / "bev_config.json")
    for k in ("x_min", "x_max", "y_min", "y_max", "H_pixel_to_world"):
        assert k in cfg, f"missing {k}"
    assert cfg["x_max"] > cfg["x_min"]
    assert cfg["y_max"] > cfg["y_min"]


def test_grid_config_exists_and_parses():
    cfg = _load_json(CONFIGS / "GITI_grid_config.json")
    assert "corners" in cfg
    assert "configuration" in cfg
    assert cfg["configuration"]["cell_size"] > 0


def test_gate_config_exists_and_parses():
    p = CONFIGS / "gate_config.yaml"
    assert p.exists()
    cfg = yaml.safe_load(p.read_text())
    assert isinstance(cfg, dict)


def test_site_configs_exist_for_giti_and_mrc():
    for site in ("giti", "mrc"):
        site_dir = CONFIGS / "sites" / site
        assert site_dir.exists(), f"missing configs/sites/{site}"
        for f in ("bev_config.json", "grid_config.json", "gate_config.yaml"):
            assert (site_dir / f).exists(), f"missing {site}/{f}"


def test_site_configs_have_expected_keys():
    for site in ("giti", "mrc"):
        bev = _load_json(CONFIGS / "sites" / site / "bev_config.json")
        grid = _load_json(CONFIGS / "sites" / site / "grid_config.json")
        gate = yaml.safe_load((CONFIGS / "sites" / site / "gate_config.yaml").read_text())
        assert "H_pixel_to_world" in bev
        assert "configuration" in grid
        assert isinstance(gate, dict)


def test_camera_matrices_are_3x3():
    import numpy as np
    for name in ("camera_matrix.npy", "camera_matrix_video_est.npy"):
        p = CONFIGS / name
        if p.exists():
            arr = np.load(p)
            assert arr.shape == (3, 3)


def test_distortion_coeffs_shape():
    import numpy as np
    for name in ("distortion_coeffs.npy", "distortion_coeffs_video_est.npy"):
        p = CONFIGS / name
        if p.exists():
            arr = np.load(p)
            assert arr.shape == (1, 5)
