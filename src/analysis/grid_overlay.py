"""Pixel-space grid overlay for review-strip rendering.

Pure functions: no I/O, no pandas. Given a grid config (same structure as
configs/GITI_grid_config.json) and a target cell name, returns pixel-space
geometry and draws it on a frame.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np

OUT_OF_BOUNDS = "OUT_OF_BOUNDS"


@dataclass(frozen=True)
class GridDims:
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    cell_size: int
    naming_style: str

    @property
    def n_cols(self) -> int:
        return max(1, (self.x_max - self.x_min) // self.cell_size)

    @property
    def n_rows(self) -> int:
        return max(1, (self.y_max - self.y_min) // self.cell_size)


def grid_from_config(cfg: dict) -> GridDims:
    corners = cfg["corners"]
    configuration = cfg.get("configuration", {})
    cell_size = int(configuration["cell_size"])
    naming_style = str(configuration["naming_style"])
    x_min = int(corners["top_left"][0])
    x_max = int(corners["top_right"][0])
    y_min = int(corners["top_left"][1])
    y_max = int(corners["bottom_left"][1])
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("invalid grid corners")
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")
    return GridDims(x_min, x_max, y_min, y_max, cell_size, naming_style)


def _col_to_letters(col_idx: int) -> str:
    letters = ""
    col_idx += 1
    while col_idx > 0:
        col_idx, remainder = divmod(col_idx - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def _letters_to_col(letters: str) -> int:
    col_idx = 0
    for ch in letters.upper():
        col_idx = col_idx * 26 + (ord(ch) - 64)
    return col_idx - 1


def cell_name(col_idx: int, row_num: int, naming_style: str) -> str:
    """Format a cell name. row_num is 1-based, col_idx is 0-based."""
    return naming_style.format(col=_col_to_letters(col_idx), row=row_num)


def pixel_to_cell(px: float, py: float, dims: GridDims) -> str:
    """Return the cell name containing the pixel, or OUT_OF_BOUNDS.

    Row numbering is 1-based, matching SpatialGrid.get_cell_from_pixels in
    src/analysis/grid_trajectory/spatial_grid.py. Column letters are Excel-style
    starting at A for column 0.
    """
    if px < dims.x_min or px >= dims.x_max:
        return OUT_OF_BOUNDS
    if py < dims.y_min or py >= dims.y_max:
        return OUT_OF_BOUNDS
    col_idx = int((px - dims.x_min) // dims.cell_size)
    row_idx_0 = int((py - dims.y_min) // dims.cell_size)
    return cell_name(col_idx, row_idx_0 + 1, dims.naming_style)


def parse_cell_name(name: str, naming_style: str) -> tuple[int, int] | None:
    """Parse a cell name into (col_idx, row_idx). Returns None on mismatch."""
    parts = re.split(r"(\{col\}|\{row\})", naming_style)
    rx_parts = []
    for part in parts:
        if part == "{col}":
            rx_parts.append(r"([A-Za-z]+)")
        elif part == "{row}":
            rx_parts.append(r"(\d+)")
        else:
            rx_parts.append(re.escape(part))
    rx = re.compile("^" + "".join(rx_parts) + "$")
    m = rx.match(name)
    if not m:
        return None
    groups = m.groups()
    col_letters = groups[0]
    row_str = groups[1]
    return _letters_to_col(col_letters), int(row_str)


def cell_bounds_px(col_idx: int, row_num: int, dims: GridDims) -> tuple[int, int, int, int]:
    """Return (x1, y1, x2, y2) for the cell in pixel space.

    col_idx is 0-based; row_num is 1-based (matches SpatialGrid).
    """
    row_idx = row_num - 1
    x1 = dims.x_min + col_idx * dims.cell_size
    y1 = dims.y_min + row_idx * dims.cell_size
    x2 = x1 + dims.cell_size
    y2 = y1 + dims.cell_size
    return (x1, y1, x2, y2)


def draw_grid_lines(
    frame: np.ndarray,
    dims: GridDims,
    color: tuple[int, int, int] = (90, 90, 90),
    thickness: int = 1,
) -> np.ndarray:
    """Draw grid lines onto a copy of the frame."""
    out = frame.copy()
    for c in range(dims.n_cols + 1):
        x = dims.x_min + c * dims.cell_size
        cv2.line(out, (x, dims.y_min), (x, dims.y_max), color, thickness)
    for r in range(dims.n_rows + 1):
        y = dims.y_min + r * dims.cell_size
        cv2.line(out, (dims.x_min, y), (dims.x_max, y), color, thickness)
    return out


def highlight_cell(
    frame: np.ndarray,
    cell: str,
    dims: GridDims,
    color: tuple[int, int, int] = (0, 255, 255),
    alpha: float = 0.30,
) -> np.ndarray:
    """Overlay a translucent highlight over the given cell."""
    parsed = parse_cell_name(cell, dims.naming_style)
    if parsed is None:
        return frame
    col_idx, row_idx = parsed
    x1, y1, x2, y2 = cell_bounds_px(col_idx, row_idx, dims)
    if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
        return frame
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def cell_center_px(col_idx: int, row_num: int, dims: GridDims) -> tuple[int, int]:
    """Return the pixel center of the cell. row_num is 1-based."""
    x1, y1, x2, y2 = cell_bounds_px(col_idx, row_num, dims)
    return ((x1 + x2) // 2, (y1 + y2) // 2)
