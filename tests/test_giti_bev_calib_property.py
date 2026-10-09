"""Property tests for load_giti_homography validators."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.bev import giti_bev_calib as gbc


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        gbc.load_giti_homography(tmp_path / "missing.json")


def test_non_positive_ransac_threshold_raises(tmp_path):
    with pytest.raises(ValueError):
        gbc.load_giti_homography(tmp_path / "x.json", ransac_thresh=0.0)
    with pytest.raises(ValueError):
        gbc.load_giti_homography(tmp_path / "x.json", ransac_thresh=-1.0)


def test_malformed_json_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not valid json")
    with pytest.raises(ValueError):
        gbc.load_giti_homography(p)


def test_too_few_points_raises(tmp_path):
    p = tmp_path / "few.json"
    p.write_text(
        json.dumps(
            {
                "calibration_points": [
                    {"pixel": {"x": 0, "y": 0}, "world": {"easting": 0.0, "northing": 0.0}},
                    {"pixel": {"x": 100, "y": 0}, "world": {"easting": 10.0, "northing": 0.0}},
                    {"pixel": {"x": 0, "y": 100}, "world": {"easting": 0.0, "northing": 10.0}},
                ]
            }
        )
    )
    with pytest.raises(ValueError):
        gbc.load_giti_homography(p)


def test_missing_points_raises(tmp_path):
    p = tmp_path / "empty.json"
    p.write_text(json.dumps({"calibration_points": []}))
    with pytest.raises(ValueError):
        gbc.load_giti_homography(p)


def test_invalid_point_entry_raises(tmp_path):
    p = tmp_path / "badpoint.json"
    p.write_text(
        json.dumps(
            {
                "calibration_points": [
                    {"pixel": {"x": 0, "y": 0}, "world": {"easting": 0.0, "northing": 0.0}},
                    {"pixel": {"x": 100, "y": 0}, "world": {"easting": 10.0, "northing": 0.0}},
                    {"pixel": {"x": 0, "y": 100}, "world": {"easting": 0.0, "northing": 10.0}},
                    {"pixel": {"x": 100, "y": 100}, "world": {"easting": 10.0, "northing": 10.0}},
                    {"missing_pixel": "bad"},
                ]
            }
        )
    )
    with pytest.raises(ValueError):
        gbc.load_giti_homography(p)


def test_valid_rectangle_gives_homography(tmp_path):
    # Four points mapping a rectangle in pixel space to a rectangle in world space
    pts = []
    for (px, py), (X, Y) in [
        ((0, 0), (0.0, 0.0)),
        ((100, 0), (10.0, 0.0)),
        ((0, 100), (0.0, 10.0)),
        ((100, 100), (10.0, 10.0)),
    ]:
        pts.append({"pixel": {"x": px, "y": py}, "world": {"easting": X, "northing": Y}})
    p = tmp_path / "ok.json"
    p.write_text(json.dumps({"calibration_points": pts}))

    H, mask, pts_world = gbc.load_giti_homography(p)
    assert H.shape == (3, 3)
    assert pts_world.shape == (4, 2)
    assert mask is not None
