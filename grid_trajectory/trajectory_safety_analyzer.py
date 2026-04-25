from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


# ---------------------------------------------------------------------
# Trajectory logging + smoothing + velocity
# ---------------------------------------------------------------------

class TrajectoryLogger:
    """
    Collects per-object, per-frame occupancy and (optionally) world coordinates,
    then converts them into time intervals per grid cell.

    Features:
    - Savitzky–Golay smoothing (optional)
    - Velocity computation (vx, vy, speed)
    - Basic velocity outlier rejection
    """

    def __init__(self, fps: float,
                 max_speed_mps: float = 40.0,
                 min_samples_per_interval: int = 3):
        self.fps = float(fps)
        self.max_speed_mps = float(max_speed_mps)
        self.min_samples_per_interval = int(min_samples_per_interval)
        # track_id:int -> list of (frame_idx:int, cell_id:any, world_x:float|None, world_y:float|None)
        self.tracks: Dict[int, List[Tuple[int, Any, Optional[float], Optional[float]]]] = defaultdict(list)

    def log(self, track_id, frame_idx, cell_id, world_x=None, world_y=None):
        tid = int(track_id)
        fi = int(frame_idx)
        self.tracks[tid].append((fi, cell_id, world_x, world_y))

    def _smooth_world_coords(self, frames: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                             window_length: int = 7, polyorder: int = 2):
        """
        Apply Savitzky–Golay smoothing to world coordinates.

        window_length must be odd and <= len(xs).
        """
        if len(xs) < window_length:
            # Not enough points to smooth; return original
            return xs, ys

        if window_length % 2 == 0:
            window_length += 1  # enforce odd

        xs_smooth = savgol_filter(xs, window_length=window_length, polyorder=polyorder)
        ys_smooth = savgol_filter(ys, window_length=window_length, polyorder=polyorder)
        return xs_smooth, ys_smooth

    def _compute_velocities(self, ts: np.ndarray, xs: np.ndarray, ys: np.ndarray):
        """
        Compute per-step velocities from world coordinates.

        Returns:
            vx, vy, speed arrays of same length as ts.
        """
        if len(ts) < 2:
            return np.zeros_like(ts), np.zeros_like(ts), np.zeros_like(ts)

        dt = np.diff(ts)
        dt[dt == 0] = 1e-6  # avoid division by zero

        vx = np.zeros_like(ts)
        vy = np.zeros_like(ts)
        speed = np.zeros_like(ts)

        dx = np.diff(xs)
        dy = np.diff(ys)
        vx[1:] = dx / dt
        vy[1:] = dy / dt
        speed[1:] = np.sqrt(vx[1:] ** 2 + vy[1:] ** 2)
        return vx, vy, speed

    def _reject_velocity_outliers(self, vx: np.ndarray, vy: np.ndarray, speed: np.ndarray):
        """
        Zero out samples where speed exceeds max_speed_mps.
        """
        mask = speed > self.max_speed_mps
        vx[mask] = 0.0
        vy[mask] = 0.0
        speed[mask] = 0.0
        return vx, vy, speed

    def build_intervals(self, smooth_world: bool = True,
                        sg_window: int = 7, sg_poly: int = 2):
        """
        Build contiguous time intervals per object per cell, optionally smoothing
        world coordinates and attaching velocity info.

        Returns:
            List[dict] with keys:
            - obj_id
            - cell_id
            - t_enter
            - t_exit
            - world_samples: list of (t, x, y, vx, vy, speed)
        """
        intervals = []

        for obj_id, samples in self.tracks.items():
            # sort by frame index
            samples.sort(key=lambda x: x[0])

            # Extract per-object arrays
            obj_frames = np.array([s[0] for s in samples], dtype=float)
            obj_ts = obj_frames / self.fps
            obj_xs = np.array(
                [float(s[2]) if s[2] is not None else np.nan for s in samples],
                dtype=float
            )
            obj_ys = np.array(
                [float(s[3]) if s[3] is not None else np.nan for s in samples],
                dtype=float
            )

            # Only smooth where we have valid coordinates
            valid = ~np.isnan(obj_xs) & ~np.isnan(obj_ys)
            if smooth_world and valid.sum() >= 3:
                xs_valid = obj_xs[valid]
                ys_valid = obj_ys[valid]
                xs_s, ys_s = self._smooth_world_coords(
                    obj_frames[valid], xs_valid, ys_valid,
                    window_length=sg_window, polyorder=sg_poly
                )
                obj_xs[valid] = xs_s
                obj_ys[valid] = ys_s

            # Compute velocities (per object)
            vx, vy, speed = self._compute_velocities(obj_ts, obj_xs, obj_ys)
            vx, vy, speed = self._reject_velocity_outliers(vx, vy, speed)

            # Build intervals using possibly smoothed coords and velocities
            prev_cell = None
            start_frame = None
            world_samples: List[Tuple[float, float, float, float, float, float]] = []

            for idx, (frame_idx, cell_id, wx, wy) in enumerate(samples):
                fi = int(frame_idx)
                t = obj_ts[idx]
                x = obj_xs[idx]
                y = obj_ys[idx]
                vxi = vx[idx]
                vyi = vy[idx]
                si = speed[idx]

                has_world = not np.isnan(x) and not np.isnan(y)

                if prev_cell is None:
                    prev_cell = cell_id
                    start_frame = fi
                    if has_world:
                        world_samples.append((t, x, y, vxi, vyi, si))
                    continue

                if cell_id == prev_cell:
                    if has_world:
                        world_samples.append((t, x, y, vxi, vyi, si))
                else:
                    # Close previous interval
                    if len(world_samples) >= self.min_samples_per_interval:
                        end_frame = fi - 1
                        intervals.append(dict(
                            obj_id=obj_id,
                            cell_id=prev_cell,
                            t_enter=start_frame / self.fps,
                            t_exit=end_frame / self.fps,
                            world_samples=world_samples,
                        ))
                    # Start new
                    prev_cell = cell_id
                    start_frame = fi
                    world_samples = []
                    if has_world:
                        world_samples.append((t, x, y, vxi, vyi, si))

            # Close final interval
            if prev_cell is not None and start_frame is not None and len(world_samples) >= self.min_samples_per_interval:
                end_frame = samples[-1][0]
                intervals.append(dict(
                    obj_id=obj_id,
                    cell_id=prev_cell,
                    t_enter=start_frame / self.fps,
                    t_exit=end_frame / self.fps,
                    world_samples=world_samples,
                ))

        return intervals


# ---------------------------------------------------------------------
# Conflict detection: PET + TTC + DRAC + neighbors
# ---------------------------------------------------------------------

class ConflictDetector:
    """
    Detects conflict pairs between intervals, computing:
    - PET
    - TTC (with interpolation at PET moment)
    - DRAC
    - (Optionally) neighbor-cell conflicts via grid_adjacency.
    """

    def __init__(self,
                 pet_threshold: float = 2.0,
                 ttc_threshold: float = 5.0,
                 collision_distance: float = 1.0):
        self.pet_threshold = pet_threshold
        self.ttc_threshold = ttc_threshold
        self.collision_distance = collision_distance

    # ---- Interpolation helpers ----

    def _interpolate_at_time(self, trajectory: List[Tuple[float, float, float, float, float, float]],
                             target_time: float):
        """Linear interpolation between samples (t, x, y, vx, vy, speed)."""
        if len(trajectory) == 0:
            return None

        for i in range(len(trajectory) - 1):
            t0, x0, y0, vx0, vy0, _ = trajectory[i]
            t1, x1, y1, vx1, vy1, _ = trajectory[i + 1]

            if t0 <= target_time <= t1:
                if t1 == t0:
                    return {'t': t0, 'x': x0, 'y': y0, 'vx': vx0, 'vy': vy0}

                alpha = (target_time - t0) / (t1 - t0)
                x = x0 + alpha * (x1 - x0)
                y = y0 + alpha * (y1 - y0)
                vx = vx0 + alpha * (vx1 - vx0)
                vy = vy0 + alpha * (vy1 - vy0)
                return {'t': target_time, 'x': x, 'y': y, 'vx': vx, 'vy': vy}

        return None

    def _compute_ttc_at_conflict(self,
                                 traj_i: List[Tuple[float, float, float, float, float, float]],
                                 traj_j: List[Tuple[float, float, float, float, float, float]],
                                 pet_moment: float):
        """
        Compute TTC at the exact PET moment using interpolation.
        pet_moment = t_entry_j (when second vehicle enters).
        """
        pos_i = self._interpolate_at_time(traj_i, pet_moment)
        pos_j = self._interpolate_at_time(traj_j, pet_moment)

        if not pos_i or not pos_j:
            return None

        # Relative kinematics at PET moment
        rx = pos_j['x'] - pos_i['x']
        ry = pos_j['y'] - pos_i['y']
        rvx = pos_j['vx'] - pos_i['vx']
        rvy = pos_j['vy'] - pos_i['vy']

        a = rvx ** 2 + rvy ** 2
        b = 2 * (rx * rvx + ry * rvy)
        c = rx ** 2 + ry ** 2 - self.collision_distance ** 2

        if a < 1e-6:
            return None

        disc = b ** 2 - 4 * a * c
        if disc < 0:
            return None

        ttc = (-b - np.sqrt(disc)) / (2 * a)
        if ttc <= 0:
            return None
        return ttc

    def _compute_drac(self,
                      traj_i: List[Tuple[float, float, float, float, float, float]],
                      traj_j: List[Tuple[float, float, float, float, float, float]],
                      ttc: float):
        """
        DRAC = required deceleration to avoid collision.
        For rear-end type approximations: DRAC = relative_speed / TTC
        """
        if ttc is None or ttc <= 0:
            return None

        pos_i = self._interpolate_at_time(traj_i, ttc)
        pos_j = self._interpolate_at_time(traj_j, ttc)
        if not pos_i or not pos_j:
            return None

        rel_vx = pos_j['vx'] - pos_i['vx']
        rel_vy = pos_j['vy'] - pos_i['vy']
        rel_speed = np.hypot(rel_vx, rel_vy)

        drac = rel_speed / ttc
        return drac

    # ---- Core conflict computation ----

    def _compute_cell_conflicts(self,
                                cell_id: Any,
                                idx_and_intervals: List[Tuple[int, Dict[str, Any]]]):
        """
        PET + TTC + DRAC for all ordered pairs of intervals within a given cell.
        """
        events = []
        n = len(idx_and_intervals)

        for i in range(n):
            idx_i, A = idx_and_intervals[i]
            for j in range(n):
                if i == j:
                    continue
                idx_j, B = idx_and_intervals[j]

                # PET logic: A leaves before B enters
                if A["t_exit"] <= B["t_enter"]:
                    pet = B["t_enter"] - A["t_exit"]
                    if 0 < pet <= self.pet_threshold:
                        traj_i = A.get("world_samples", [])
                        traj_j = B.get("world_samples", [])

                        # TTC at PET moment
                        ttc = self._compute_ttc_at_conflict(traj_i, traj_j, B["t_enter"])

                        # DRAC
                        drac = self._compute_drac(traj_i, traj_j, ttc) if ttc is not None else None

                        events.append(dict(
                            obj_i=A["obj_id"],
                            obj_j=B["obj_id"],
                            cell_id=cell_id,
                            t_exit_i=A["t_exit"],
                            t_entry_j=B["t_enter"],
                            PET=pet,
                            TTC=ttc,
                            DRAC=drac,
                            duration_i=A["t_exit"] - A["t_enter"],
                            duration_j=B["t_exit"] - B["t_enter"],
                            world_traj_i=traj_i,
                            world_traj_j=traj_j,
                            interval_i_idx=idx_i,
                            interval_j_idx=idx_j,
                        ))

        return events

    def compute_conflicts(self,
                          intervals: List[Dict[str, Any]],
                          grid_adjacency: Optional[Dict[Any, List[Any]]] = None):
        """
        Compute PET + TTC + DRAC conflict events between intervals.

        Args:
            intervals: list of interval dicts from TrajectoryLogger.build_intervals().
            grid_adjacency: optional dict mapping cell_id -> list of neighboring cell_ids
                            to also consider cross-cell conflicts.

        Returns:
            List[dict] with conflict events.
        """
        events = []
        by_cell: Dict[Any, List[Tuple[int, Dict[str, Any]]]] = {}

        # Group intervals by cell
        for idx, iv in enumerate(intervals):
            cell_id = iv["cell_id"]
            by_cell.setdefault(cell_id, []).append((idx, iv))

        # 1) Same-cell conflicts
        for cell_id, idx_and_intervals in by_cell.items():
            events.extend(self._compute_cell_conflicts(cell_id, idx_and_intervals))

        # 2) Neighbor-cell conflicts (optional)
        if grid_adjacency:
            for cell_id, neighbors in grid_adjacency.items():
                if cell_id not in by_cell:
                    continue
                for nb in neighbors:
                    if nb not in by_cell:
                        continue
                    pairs = []
                    for idx_i, iv_i in by_cell[cell_id]:
                        pairs.append((idx_i, iv_i))
                    for idx_j, iv_j in by_cell[nb]:
                        pairs.append((idx_j, iv_j))
                    # Treat cell_id/neighbor pair as a combined "virtual cell"
                    events.extend(self._compute_cell_conflicts(
                        cell_id=f"{cell_id}__{nb}",
                        idx_and_intervals=pairs
                    ))

        return events


# ---------------------------------------------------------------------
# Example end-to-end runner (adapt column names)
# ---------------------------------------------------------------------

def run_pet_ttc_drac_pipeline(csv_path: str,
                              fps: float = 30.0,
                              pet_threshold: float = 2.0,
                              ttc_threshold: float = 5.0,
                              smooth_world: bool = True,
                              max_speed_mps: float = 40.0,
                              min_samples_per_interval: int = 3,
                              grid_adjacency: Optional[Dict[Any, List[Any]]] = None):
    """
    Example end-to-end: read BEV events CSV, build intervals, compute PET+TTC+DRAC.
    """
    df = pd.read_csv(csv_path)

    # Adapt these column names to match your CSV schema
    TRACK_COL = "track_id"
    FRAME_COL = "frame_idx"
    CELL_COL = "cell_id"
    WX_COL = "world_x"
    WY_COL = "world_y"

    logger = TrajectoryLogger(
        fps=fps,
        max_speed_mps=max_speed_mps,
        min_samples_per_interval=min_samples_per_interval,
    )

    for _, row in df.iterrows():
        logger.log(
            track_id=row[TRACK_COL],
            frame_idx=row[FRAME_COL],
            cell_id=row[CELL_COL],
            world_x=row.get(WX_COL, None),
            world_y=row.get(WY_COL, None),
        )

    intervals = logger.build_intervals(
        smooth_world=smooth_world,
        sg_window=7,
        sg_poly=2,
    )

    detector = ConflictDetector(
        pet_threshold=pet_threshold,
        ttc_threshold=ttc_threshold,
        collision_distance=1.0,
    )
    conflicts = detector.compute_conflicts(intervals, grid_adjacency=grid_adjacency)

    return intervals, conflicts
