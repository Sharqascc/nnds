import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml  # ensure PyYAML is in requirements.txt

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
DOCS_DIR = ROOT / "docs"
DEMO_CSV = DOCS_DIR / "data_samples" / "petevents_bev_demo.csv"


# ---------------------------
# Helpers
# ---------------------------

def _iter_json_configs():
    if not CONFIG_DIR.exists():
        return []
    return sorted(CONFIG_DIR.glob("*.json"))


def _load_json(path: Path):
    text = path.read_text()
    try:
        data = json.loads(text)
    except Exception as exc:
        raise AssertionError(f"Failed to parse JSON config: {path}") from exc
    assert isinstance(
        data, (dict, list)
    ), f"Config {path} must be a JSON object or array, got {type(data).__name__}"
    return data


# ---------------------------
# Generic JSON existence & parse tests
# ---------------------------

def test_config_dir_exists():
    assert CONFIG_DIR.exists(), f"Config dir not found: {CONFIG_DIR}"


def test_all_json_configs_parse():
    json_files = _iter_json_configs()
    assert json_files, "No .json files found in configs/"

    for cfg in json_files:
        _ = _load_json(cfg)


# ---------------------------
# YAML support: gate_config.yaml
# ---------------------------

def test_gate_config_yaml_parses_and_has_basic_keys():
    gate_path = CONFIG_DIR / "gate_config.yaml"
    assert gate_path.exists(), f"Missing gate_config.yaml at {gate_path}"

    text = gate_path.read_text()
    try:
        data = yaml.safe_load(text)
    except Exception as exc:
        raise AssertionError(f"Failed to parse YAML config: {gate_path}") from exc

    assert isinstance(data, dict), f"gate_config.yaml must be a mapping, got {type(data).__name__}"

    # Minimal schema: expect a top-level gates list or mapping
    assert "gates" in data, "gate_config.yaml must define a top-level 'gates' key"
    gates = data["gates"]
    assert isinstance(
        gates, (list, dict)
    ), f"'gates' must be a list or dict, got {type(gates).__name__}"
    assert gates, "gate_config.yaml 'gates' collection must not be empty"


# ---------------------------
# Schema & bounds checks: bev_config.json
# ---------------------------

def test_bev_config_has_required_fields_and_bounds():
    bev_cfg_path = CONFIG_DIR / "bev_config.json"
    assert bev_cfg_path.exists(), f"Missing bev_config.json at {bev_cfg_path}"

    cfg = _load_json(bev_cfg_path)
    assert isinstance(cfg, dict), "bev_config.json must be a JSON object"

    # New schema: bev_bounds + bev_resolution
    bev_bounds = cfg.get("bev_bounds")
    assert isinstance(bev_bounds, dict), "bev_config.json must have 'bev_bounds' object"

    for key in ["x_min", "x_max", "y_min", "y_max"]:
        assert key in bev_bounds, f"bev_bounds missing required key: '{key}'"

    xmin = float(bev_bounds["x_min"])
    xmax = float(bev_bounds["x_max"])
    ymin = float(bev_bounds["y_min"])
    ymax = float(bev_bounds["y_max"])

    resolution = cfg.get("bev_resolution")
    assert resolution is not None, "bev_config.json must define 'bev_resolution'"

    # Basic bounds sanity
    assert xmax > xmin, f"BEV x_max must be > x_min, got x_min={xmin}, x_max={xmax}"
    assert ymax > ymin, f"BEV y_max must be > y_min, got y_min={ymin}, y_max={ymax}"

    # Resolution should be positive and reasonable [w, h]
    if isinstance(resolution, (int, float)):
        assert resolution > 0, f"BEV resolution must be positive, got {resolution}"
    elif isinstance(resolution, (list, tuple)):
        assert len(resolution) == 2, f"bev_resolution must be [width, height], got {len(resolution)}"
        assert all(
            isinstance(v, (int, float)) and v > 0 for v in resolution
        ), f"bev_resolution entries must be positive numbers, got {resolution}"
    else:
        raise AssertionError(f"Unexpected type for bev_resolution: {type(resolution).__name__}")

    # CRITICAL: Check that BEV bounds span a reasonable intersection area
    x_span = xmax - xmin
    y_span = ymax - ymin
    if x_span < 10 or y_span < 10:
        warnings.warn(
            (
                f"BEV bounds too small ({x_span:.2f}m x {y_span:.2f}m). "
                "Intersection should typically span >10m in each direction."
            ),
            UserWarning,
        )


def test_bev_config_has_valid_homography():
    """Optional: check presence and basic quality of homography if stored in config."""
    bev_cfg_path = CONFIG_DIR / "bev_config.json"
    cfg = _load_json(bev_cfg_path)

    H = cfg.get("H_pixel_to_world")
    if H is None:
        # It's OK if H is computed elsewhere (e.g., from calibration points)
        return

    assert len(H) == 3, "Homography must be 3x3 (3 rows)"
    assert all(len(row) == 3 for row in H), "Homography must be 3x3 (3 columns per row)"

    H_array = np.array(H, dtype=float)
    if np.allclose(H_array, np.eye(3)):
        warnings.warn(
            "Homography appears to be identity matrix - check calibration",
            UserWarning,
        )


# ---------------------------
# Schema & bounds checks: giti_calibration_points.json
# ---------------------------

def test_calibration_points_have_spread_and_required_fields():
    calib_path = CONFIG_DIR / "giti_calibration_points.json"
    assert calib_path.exists(), f"Missing giti_calibration_points.json at {calib_path}"

    cfg = _load_json(calib_path)
    assert isinstance(cfg, dict), "giti_calibration_points.json must be a JSON object"

    points = cfg.get("calibration_points")
    assert isinstance(points, list) and points, (
        "giti_calibration_points.json must define non-empty 'calibration_points' list"
    )

    xs = []
    ys = []

    for p in points:
        assert isinstance(p, dict), "Each calibration point must be a JSON object"
        assert "pixel" in p and "world" in p, "Each calibration point must have 'pixel' and 'world' keys"

        # pixel sanity
        pix = p["pixel"]
        assert "x" in pix and "y" in pix, "Pixel entry must contain 'x' and 'y'"

        # world sanity (local coordinates or absolute coordinates)
        w = p["world"]
        assert isinstance(w, dict), "'world' field must be an object"

        has_local = "x_m" in w and "y_m" in w
        has_global = "easting" in w and "northing" in w
        assert has_local or has_global, (
            "World coordinates must define either ('x_m','y_m') or ('easting','northing')"
        )

        if has_local:
            xs.append(float(w["x_m"]))
            ys.append(float(w["y_m"]))
        else:
            xs.append(float(w["easting"]))
            ys.append(float(w["northing"]))

    if xs and ys:
        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)
        # Recommended minimum spread for a real intersection
        min_spread = 15.0  # meters (or equivalent units)
        if x_range < min_spread and y_range < min_spread:
            warnings.warn(
                (
                    f"Calibration points span only {x_range:.1f} x {y_range:.1f} units; "
                    f"expected at least ~{min_spread} in at least one direction."
                ),
                UserWarning,
            )


# ---------------------------
# Demo CSV presence & data quality
# ---------------------------

def test_demo_csv_exists_and_has_rows():
    assert DEMO_CSV.exists(), f"Demo PET CSV not found at {DEMO_CSV}"

    df = pd.read_csv(DEMO_CSV)
    assert not df.empty, f"Demo PET CSV at {DEMO_CSV} is empty"

    required_cols = {"event_id", "pet", "track_a", "track_b"}
    missing = required_cols - set(df.columns)
    assert not missing, f"Demo PET CSV missing required columns: {sorted(missing)}"

    # Check for actual data (not just headers/NaNs)
    if df["event_id"].isna().all():
        warnings.warn(
            (
                f"Demo PET CSV at {DEMO_CSV} contains only NaN values (headers-only or placeholder rows). "
                "Consider populating with real sample data."
            ),
            UserWarning,
        )
    else:
        # PET should be strictly positive
        if (df["pet"] <= 0).any():
            warnings.warn(
                (
                    f"Demo PET CSV at {DEMO_CSV} contains non-positive PET values (<=0). "
                    "PET should be strictly positive."
                ),
                UserWarning,
            )
