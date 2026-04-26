from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, List

import cv2
import numpy as np


OUT_OF_BOUNDS_CELL = "OUT_OF_BOUNDS"


@dataclass
class GridConfig:
    corners: Dict[str, Tuple[int, int]]
    cell_size: int
    naming_style: str


class SpatialGrid:
    """
    Pixel-space grid for mapping image coordinates to intersection cells.

    Expected config JSON structure (GITI_grid_config.json style):
        {
            "corners": {
                "top_left": [x, y],
                "top_right": [x, y],
                "bottom_left": [x, y],
                "bottom_right": [x, y]
            },
            "configuration": {
                "cell_size": 40,
                "naming_style": "G_{col}_{row}"
            }
        }
    """

    def __init__(self, config_path: Union[str, Path]) -> None:
        config_path = Path(config_path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Grid config not found: {config_path}")

        with config_path.open("r") as f:
            raw_cfg: Dict[str, Any] = json.load(f)

        corners = raw_cfg.get("corners")
        configuration = raw_cfg.get("configuration", {})

        if not isinstance(corners, dict):
            raise ValueError("Grid config must contain a 'corners' object.")

        required_corners = {"top_left", "top_right", "bottom_left", "bottom_right"}
        missing = required_corners - set(corners.keys())
        if missing:
            raise KeyError(f"Grid corners missing keys: {sorted(missing)}")

        cell_size = configuration.get("cell_size")
        naming_style = configuration.get("naming_style")

        if not isinstance(cell_size, int) or cell_size <= 0:
            raise ValueError(f"cell_size must be a positive int, got {cell_size!r}")
        if not isinstance(naming_style, str) or "{col}" not in naming_style or "{row}" not in naming_style:
            raise ValueError(
                f"naming_style must be a format string with {{col}} and {{row}}, got {naming_style!r}"
            )

        self.config = GridConfig(
            corners={k: (int(v[0]), int(v[1])) for k, v in corners.items()},
            cell_size=cell_size,
            naming_style=naming_style,
        )

        # Derived attributes
        self.corners = self.config.corners
        self.cell_size = self.config.cell_size
        self.naming_style = self.config.naming_style

        # Boundaries for coordinate checking
        self.x_min, self.x_max = self.corners["top_left"][0], self.corners["top_right"][0]
        self.y_min, self.y_max = self.corners["top_left"][1], self.corners["bottom_left"][1]

        # Cache for cell centers (minor perf optimization)
        self._cell_center_cache: Dict[str, Optional[Tuple[int, int]]] = {}

    def __repr__(self) -> str:
        return (
            f"SpatialGrid(cell_size={self.cell_size}, "
            f"x_range=({self.x_min}, {self.x_max}), "
            f"y_range=({self.y_min}, {self.y_max}))"
        )

    def get_cell_from_pixels(self, px_x: float, px_y: float) -> str:
        """
        Convert (x, y) pixels into a Cell ID (e.g., G_B_3).

        Returns:
            Cell ID string or OUT_OF_BOUNDS_CELL if outside grid.
        """
        if not (self.x_min <= px_x <= self.x_max and self.y_min <= px_y <= self.y_max):
            return OUT_OF_BOUNDS_CELL

        col_idx = int((px_x - self.x_min) // self.cell_size)
        row_idx = int((px_y - self.y_min) // self.cell_size)

        col_idx = max(col_idx, 0)
        row_idx = max(row_idx, 0)

        col_letter = chr(65 + (col_idx % 26))
        row_num = row_idx + 1

        return self.naming_style.format(col=col_letter, row=row_num)

    def get_cell_center(self, cell_id: str) -> Optional[Tuple[int, int]]:
        """
        Reverse a Cell ID back into pixel coordinates (center of the cell).

        Returns None if the cell_id is malformed.
        Uses a small cache to avoid recomputing for repeated lookups.
        """
        if not isinstance(cell_id, str):
            return None

        if cell_id in self._cell_center_cache:
            return self._cell_center_cache[cell_id]

        try:
            parts = cell_id.split("_")
            # Expected format like: G_A_3 (prefix, col, row)
            if len(parts) != 3:
                result = None
            else:
                col_letter = parts[1]
                row_num = int(parts[2])

                if len(col_letter) != 1 or not col_letter.isalpha():
                    result = None
                else:
                    col_idx = ord(col_letter.upper()) - 65
                    row_idx = row_num - 1

                    if col_idx < 0 or row_idx < 0:
                        result = None
                    else:
                        x = self.x_min + (col_idx * self.cell_size) + (self.cell_size // 2)
                        y = self.y_min + (row_idx * self.cell_size) + (self.cell_size // 2)
                        result = (int(x), int(y))
        except (ValueError, IndexError):
            result = None

        self._cell_center_cache[cell_id] = result
        return result

    def get_cell_bounds(self, cell_id: str) -> Optional[Tuple[int, int, int, int]]:
        """
        Get bounding box (x1, y1, x2, y2) of a cell in pixel coordinates.

        Useful for drawing a highlight rectangle over a specific cell.
        """
        center = self.get_cell_center(cell_id)
        if center is None:
            return None

        cx, cy = center
        half = self.cell_size // 2
        return (cx - half, cy - half, cx + half, cy + half)

    def get_stats(self) -> Dict[str, Any]:
        """
        Return basic grid statistics for debugging / logging.
        """
        n_cols = (self.x_max - self.x_min) // self.cell_size
        n_rows = (self.y_max - self.y_min) // self.cell_size
        return {
            "cell_size": self.cell_size,
            "n_cols": n_cols,
            "n_rows": n_rows,
            "total_cells": n_cols * n_rows,
            "x_range": (self.x_min, self.x_max),
            "y_range": (self.y_min, self.y_max),
        }

    def draw_overlay(
        self,
        frame: np.ndarray,
        alpha: float = 0.6,
        line_color: Tuple[int, int, int] = (0, 255, 255),  # Neon yellow
        text_color: Tuple[int, int, int] = (0, 255, 255),
        highlight_cells: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Render a high-visibility grid with axis labels over the frame.

        Args:
            frame: BGR image (H, W, 3).
            alpha: Blend factor for overlay (0–1).
            line_color: Color for grid lines (B, G, R).
            text_color: Color for text labels (B, G, R).
            highlight_cells: Optional list of cell IDs to highlight.

        Returns:
            New frame with grid overlay drawn.
        """
        overlay = frame.copy()
        shadow = (0, 0, 0)

        # 1. Draw vertical lines & column headers (A, B, C...)
        for i, x in enumerate(range(self.x_min, self.x_max + 1, self.cell_size)):
            cv2.line(overlay, (x, self.y_min), (x, self.y_max), shadow, 3)
            cv2.line(overlay, (x, self.y_min), (x, self.y_max), line_color, 1)

            if x < self.x_max:
                label = chr(65 + (i % 26))
                pos = (x + 10, max(30, self.y_min - 15))
                cv2.putText(overlay, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, shadow, 4)
                cv2.putText(overlay, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

        # 2. Draw horizontal lines & row headers (1, 2, 3...)
        for i, y in enumerate(range(self.y_min, self.y_max + 1, self.cell_size)):
            cv2.line(overlay, (self.x_min, y), (self.x_max, y), shadow, 3)
            cv2.line(overlay, (self.x_min, y), (self.x_max, y), line_color, 1)

            if y < self.y_max:
                label = str(i + 1)
                pos = (max(5, self.x_min - 45), y + 35)
                cv2.putText(overlay, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, shadow, 4)
                cv2.putText(overlay, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

        # 3. Optionally highlight specific cells
        if highlight_cells:
            for cid in highlight_cells:
                bounds = self.get_cell_bounds(cid)
                if bounds is None:
                    continue
                x1, y1, x2, y2 = bounds
                cv2.rectangle(
                    overlay,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),  # red highlight by default
                    2,
                )

        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0.0)
