import os

import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_pet_csv(csv_path: str) -> pd.DataFrame:
    """Load PET CSV and parse world trajectories into Python lists."""
    df = pd.read_csv(csv_path)
    df["traj_i"] = df["world_traj_i"].apply(ast.literal_eval)
    df["traj_j"] = df["world_traj_j"].apply(ast.literal_eval)
    return df


def compute_timing_from_traj(df: pd.DataFrame) -> pd.DataFrame:
    """Add approximate timing info (for visualization only) to PET dataframe."""

    def closest_approach_times(traj_i, traj_j):
        ti, xi, yi = zip(*traj_i)
        tj, xj, yj = zip(*traj_j)

        ti = np.array(ti); xi = np.array(xi); yi = np.array(yi)
        tj = np.array(tj); xj = np.array(xj); yj = np.array(yj)

        T = min(len(ti), len(tj))
        ti_c = ti[:T]; xi_c = xi[:T]; yi_c = yi[:T]
        tj_c = tj[:T]; xj_c = xj[:T]; yj_c = yj[:T]

        dist = np.hypot(xi_c - xj_c, yi_c - yj_c)
        k_min = int(np.argmin(dist))

        t_closest = float(ti_c[k_min])
        k_leave = max(0, k_min - 1)
        k_enter = min(T - 1, k_min + 1)

        t_leave_i = float(ti_c[k_leave])
        t_enter_j = float(tj_c[k_enter])

        return {
            "t_closest": t_closest,
            "t_leave_i": t_leave_i,
            "t_enter_j": t_enter_j,
            "pet_approx": float(t_enter_j - t_leave_i),
            "dist_min": float(dist[k_min]),
            "k_closest": k_min,
        }

    timing = df.apply(
        lambda r: closest_approach_times(r["traj_i"], r["traj_j"]),
        axis=1,
        result_type="expand",
    )
    return pd.concat([df, timing], axis=1)


def get_class_default(track_id: int) -> str:
    """Default class mapper: everything is just 'vehicle'."""
    return "vehicle"


def plot_conflict_event(df: pd.DataFrame,
                        event_id: int,
                        class_mapper=get_class_default,
                        save_path: str | None = None):
    """Plot BEV trajectories and PET info for a single conflict event."""
    row = df.loc[df["event_id"] == event_id].iloc[0]

    traj_i = row["traj_i"]
    traj_j = row["traj_j"]

    ti, xi, yi = zip(*traj_i)
    tj, xj, yj = zip(*traj_j)

    ti = np.array(ti); xi = np.array(xi); yi = np.array(yi)
    tj = np.array(tj); xj = np.array(xj); yj = np.array(yj)

    class_i = class_mapper(int(row["track_a"]))
    class_j = class_mapper(int(row["track_b"]))

    pet = float(row["pet"])
    pet_approx = float(row["pet_approx"])
    cell = row["conflict_type"]
    t_leave_i = float(row["t_leave_i"])
    t_enter_j = float(row["t_enter_j"])

    plt.figure(figsize=(6, 6))
    plt.plot(xi, yi, "-o",
             label=f"Track {int(row['track_a'])} ({class_i})",
             alpha=0.8)
    plt.plot(xj, yj, "-o",
             label=f"Track {int(row['track_b'])} ({class_j})",
             alpha=0.8)

    # Mark closest approach using aligned segments
    T = min(len(ti), len(tj))
    xi_c = xi[:T]; yi_c = yi[:T]
    xj_c = xj[:T]; yj_c = yj[:T]

    dist = np.hypot(xi_c - xj_c, yi_c - yj_c)
    k_closest = int(np.argmin(dist))
    plt.scatter([xi_c[k_closest]], [yi_c[k_closest]],
                c="red", marker="x", s=80, label="closest approach")

    plt.xlabel("X (world / BEV)")
    plt.ylabel("Y (world / BEV)")
    plt.title(
        f"Conflict {event_id} – {class_i} vs {class_j}\n"
        f"Cell {cell}, PET = {pet:.3f} s (approx {pet_approx:.3f} s)\n"
        f"t_leave_i = {t_leave_i:.3f}, t_enter_j = {t_enter_j:.3f}"
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.gca().set_aspect("equal", "box")
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()
