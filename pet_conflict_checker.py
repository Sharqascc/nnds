"""
PET conflict checker with improved BEV visualization and frame choice.

- Reads outputs/petevents_bev_30frames.csv
- Selects a given event_id
- Plots:
    * BEV trajectories with proper axes, units, grid, and PET annotation
    * Color-coded by time so direction and motion are visible
- Chooses a video frame near the PET moment and saves:
    * Frame with grid and conflict cell highlighted
"""

import os
import ast
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class PETCheckConfig:
    csv_path: str = "outputs/petevents_bev_30frames.csv"
    video_path: str = "videos/traffic_video.mp4"
    grid_config_path: str = "configs/GITI_grid_config.json"
    fps: float = 30.0
    out_dir: str = "outputs"
    event_id: int = 1


class SpatialGrid:
    def __init__(self, config_path: str):
        import json
        with open(config_path, "r") as f:
            cfg = json.load(f)
        self.cell_size = cfg.get("cell_size_pixels", 40)
        self.cells = cfg.get("cells", {})

    def get_cell_center(self, cell_id: str):
        info = self.cells.get(cell_id)
        if info is None:
            return None
        return info["cx"], info["cy"]

    def draw_overlay(self, frame: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        overlay = frame.copy()
        color = (0, 255, 255)
        for cell_id, info in self.cells.items():
            cx, cy = info["cx"], info["cy"]
            half = self.cell_size / 2
            x1, y1 = int(cx - half), int(cy - half)
            x2, y2 = int(cx + half), int(cy + half)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def highlight_conflict_cell_on_frame(
    frame: np.ndarray, grid: SpatialGrid, cell_id: str, color=(0, 0, 255)
) -> np.ndarray:
    center = grid.get_cell_center(cell_id)
    if center is None:
        return frame
    cx, cy = center
    half = grid.cell_size / 2
    x1, y1 = int(cx - half), int(cy - half)
    x2, y2 = int(cx + half), int(cy + half)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=-1)
    blended = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    return blended


def save_conflict_frame(
    video_path: str,
    grid_config_path: str,
    cell_id: str,
    frame_idx: int,
    out_path: str,
    alpha: float = 0.6,
) -> str:
    grid = SpatialGrid(grid_config_path)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")

    frame_grid = grid.draw_overlay(frame, alpha=alpha)
    frame_final = highlight_conflict_cell_on_frame(frame_grid, grid, cell_id)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, frame_final)
    return out_path


def load_pet_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["traj_i"] = df["world_traj_i"].apply(ast.literal_eval)
    df["traj_j"] = df["world_traj_j"].apply(ast.literal_eval)
    return df


def compute_time_window(traj_i, traj_j, pet_csv: float):
    """
    Derive a reasonable time window and visualization time for the event.

    Heuristic:
    - Take mid-time of each trajectory
    - Average them to get t_center
    - Use PET to define a window around t_center
    - Choose visualization time t_vis inside that window
    """
    ti = np.array([p[0] for p in traj_i], dtype=float)
    tj = np.array([p[0] for p in traj_j], dtype=float)

    t_mid_i = float(ti[len(ti) // 2])
    t_mid_j = float(tj[len(tj) // 2])
    t_center = 0.5 * (t_mid_i + t_mid_j)

    # Use ± PET as a soft window
    pad = max(float(pet_csv), 0.5)
    t_start = max(0.0, t_center - pad)
    t_end = t_center + pad

    # Visualization time: center of that window
    t_vis = 0.5 * (t_start + t_end)
    return t_start, t_end, t_vis


def plot_conflict_event(row: pd.Series, save_path: str = None):
    traj_i = row["traj_i"]
    traj_j = row["traj_j"]

    ti, xi, yi = zip(*traj_i)
    tj, xj, yj = zip(*traj_j)
    ti = np.array(ti); xi = np.array(xi); yi = np.array(yi)
    tj = np.array(tj); xj = np.array(xj); yj = np.array(yj)

    pet_csv = float(row["pet"])
    track_a = int(row["track_a"])
    track_b = int(row["track_b"])
    cell_id = row["conflict_type"]

    # Normalize coordinates so conflict is near origin (improves readability)
    x_all = np.concatenate([xi, xj])
    y_all = np.concatenate([yi, yj])
    x0 = 0.5 * (x_all.min() + x_all.max())
    y0 = 0.5 * (y_all.min() + y_all.max())
    xi_n = xi - x0
    yi_n = yi - y0
    xj_n = xj - x0
    yj_n = yj - y0

    # Distance for closest approach
    T = min(len(ti), len(tj))
    dist = np.hypot(xi_n[:T] - xj_n[:T], yi_n[:T] - yj_n[:T])
    k_min = int(np.argmin(dist))
    d_min = float(dist[k_min])

    fig, ax = plt.subplots(figsize=(8, 7))

    # Time-colored scatter to show direction and temporal evolution
    sc_i = ax.scatter(xi_n, yi_n, c=ti, cmap="Blues", s=24, label=f"Track {track_a}", alpha=0.8)
    sc_j = ax.scatter(xj_n, yj_n, c=tj, cmap="Oranges", s=24, label=f"Track {track_b}", alpha=0.8)

    # Draw simple lines through them for shape
    ax.plot(xi_n, yi_n, color="tab:blue", linewidth=1.0, alpha=0.6)
    ax.plot(xj_n, yj_n, color="tab:orange", linewidth=1.0, alpha=0.6)

    # Start markers
    ax.scatter(xi_n[0], yi_n[0], color="navy", marker="o", s=40, zorder=5)
    ax.scatter(xj_n[0], yj_n[0], color="darkorange", marker="o", s=40, zorder=5)

    # Closest approach marker
    ax.scatter(xi_n[k_min], yi_n[k_min], color="red", marker="X", s=120,
               edgecolor="white", linewidth=1.5, label=f"Closest (d={d_min:.2f} m)")

    # Axes and grid
    ax.set_aspect("equal", "box")
    ax.set_xlabel("X (meters, localized)", fontsize=12)
    ax.set_ylabel("Y (meters, localized)", fontsize=12)
    ax.grid(True, alpha=0.3, linestyle="--")

    # PET annotation
    ax.text(
        0.02,
        0.98,
        f"PET = {pet_csv:.3f} s | d_min = {d_min:.2f} m",
        transform=ax.transAxes,
        fontsize=12,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    ax.set_title(
        f"BEV Conflict Analysis – Event {int(row['event_id'])} ({cell_id})",
        fontsize=14,
        fontweight="bold",
    )

    # Colorbars for time
    cbar_i = fig.colorbar(sc_i, ax=ax, fraction=0.046, pad=0.04)
    cbar_i.set_label("Time (s) – Track A", fontsize=10)
    cbar_j = fig.colorbar(sc_j, ax=ax, fraction=0.046, pad=0.08)
    cbar_j.set_label("Time (s) – Track B", fontsize=10)

    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def main(cfg: PETCheckConfig):
    print("=== PET Conflict Checker (improved BEV) ===")
    print(f"CSV:   {cfg.csv_path}")
    print(f"Video: {cfg.video_path}")
    print(f"Grid:  {cfg.grid_config_path}")
    print(f"Event: {cfg.event_id}")

    df = load_pet_csv(cfg.csv_path)
    if "event_id" not in df.columns:
        raise ValueError("CSV must contain 'event_id' column")

    row = df.loc[df["event_id"] == cfg.event_id].iloc[0]

    # 1) BEV plot
    bev_out = os.path.join(cfg.out_dir, f"conflict_{cfg.event_id}_bev.png")
    print(f"Plotting BEV trajectories to {bev_out} ...")
    plot_conflict_event(row, save_path=bev_out)
    print("BEV plot done.")

    # 2) Derive visualization time window from trajectories and PET
    traj_i = row["traj_i"]
    traj_j = row["traj_j"]
    pet_csv = float(row["pet"])
    t_start, t_end, t_vis = compute_time_window(traj_i, traj_j, pet_csv)

    frame_idx = int(round(t_vis * cfg.fps))
    print(f"PET = {pet_csv:.3f} s, t_window=({t_start:.3f}, {t_end:.3f}), t_vis={t_vis:.3f}")
    print(f"Visualizing at frame index {frame_idx}")

    # 3) Save raw frame with grid and conflict cell highlight
    frame_out = os.path.join(cfg.out_dir, f"conflict_{cfg.event_id}_frame_mid.png")
    print(f"Saving conflict frame with grid/cell highlight to {frame_out} ...")
    save_conflict_frame(
        video_path=cfg.video_path,
        grid_config_path=cfg.grid_config_path,
        cell_id=row["conflict_type"],
        frame_idx=frame_idx,
        out_path=frame_out,
        alpha=0.6,
    )
    print("Frame saved.")
    print("=== Done ===")


if __name__ == "__main__":
    cfg = PETCheckConfig()
    main(cfg)
