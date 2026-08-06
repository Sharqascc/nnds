import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from traffic_diffusion.trajectory_diffusion import TrajectoryDataset, TrajectoryDiffusionModel


def parse_traj_txy(cell):
    arr = np.array(ast.literal_eval(cell), dtype=float)  # (T,3): t,x,y
    return arr[:, 0], arr[:, 1:3]


def build_training_tensors(csv_path, Th=16):
    import pandas as pd
    import numpy as np
    import torch, ast, glob
    from pathlib import Path

    csv_path = Path(csv_path)

    # Prioritize generated outputs over demo samples
    if not csv_path.exists() or "data_samples" in str(csv_path):
        candidates = sorted([Path(f) for f in glob.glob("outputs/petevents_*.csv") if not f.endswith("_detections.csv")])
        if candidates:
            csv_path = candidates[-1]
            print(f"ℹ️ Auto-selected generated PET CSV from outputs: {csv_path}")

    if not csv_path.exists():
        raise FileNotFoundError(f"Could not locate PET events CSV at {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV file {csv_path} is empty.")

    x0_list, cond_list = [], []

    # Case 1: world_traj_i / world_traj_j embedded in CSV
    if "world_traj_i" in df.columns and "world_traj_j" in df.columns:
        for ci, cj in zip(df["world_traj_i"], df["world_traj_j"]):
            ti = np.array(ast.literal_eval(ci) if isinstance(ci, str) else ci, dtype=np.float32)
            tj = np.array(ast.literal_eval(cj) if isinstance(cj, str) else cj, dtype=np.float32)
            if len(ti) >= Th and len(tj) >= Th:
                x0_list.append(ti[:Th])
                cond_list.append(tj[:Th])

    # Case 2: Reconstruct trajectories using companion detections CSV in the same folder
    else:
        det_path = csv_path.parent / f"{csv_path.stem}_detections.csv"
        if not det_path.exists():
            clean_stem = csv_path.stem.replace("petevents_bev_", "").replace("petevents_", "")
            det_path = csv_path.parent / f"{clean_stem}_detections.csv"

        if not det_path.exists():
            candidates = sorted([Path(f) for f in glob.glob(f"{csv_path.parent}/*_detections.csv")])
            if candidates:
                det_path = candidates[-1]

        if not det_path.exists():
            raise FileNotFoundError(f"Could not locate companion detections file for {csv_path}")

        print(f"✅ Reconstructing trajectories using matching detections file: {det_path}")
        det_df = pd.read_csv(det_path)

        id_col_i = next((c for c in ["track_id_i", "actor_i", "id_i", "track_a", "track_i"] if c in df.columns), None)
        id_col_j = next((c for c in ["track_id_j", "actor_j", "id_j", "track_b", "track_j"] if c in df.columns), None)

        if not id_col_i or not id_col_j:
            raise KeyError(f"Could not locate track ID columns in {csv_path}. Columns found: {list(df.columns)}")

        track_col = next((c for c in ["track_id", "actor_id", "id"] if c in det_df.columns), None)
        frame_col = next((c for c in ["frame", "frame_idx", "frame_id"] if c in det_df.columns), "frame")
        x_col = next((c for c in ["cx", "world_x", "x", "bev_x", "x1"] if c in det_df.columns), None)
        y_col = next((c for c in ["cy", "world_y", "y", "bev_y", "y1"] if c in det_df.columns), None)

        if not track_col or not x_col or not y_col:
            raise KeyError(f"Missing required trajectory columns in {det_path}. Columns found: {list(det_df.columns)}")

        for _, row in df.iterrows():
            id_i, id_j = row[id_col_i], row[id_col_j]
            traj_i = det_df[det_df[track_col] == id_i][[frame_col, x_col, y_col]].sort_values(frame_col)[[x_col, y_col]].values
            traj_j = det_df[det_df[track_col] == id_j][[frame_col, x_col, y_col]].sort_values(frame_col)[[x_col, y_col]].values

            if len(traj_i) > 0 and len(traj_j) > 0:
                if len(traj_i) < Th:
                    traj_i = np.pad(traj_i, ((0, Th - len(traj_i)), (0, 0)), mode="edge")
                if len(traj_j) < Th:
                    traj_j = np.pad(traj_j, ((0, Th - len(traj_j)), (0, 0)), mode="edge")

                x0_list.append(traj_i[:Th])
                cond_list.append(traj_j[:Th])

    if len(x0_list) == 0:
        raise ValueError(f"No valid trajectory pairs extracted from {csv_path}.")

    x0_tensor = torch.tensor(np.array(x0_list), dtype=torch.float32)
    cond_tensor = torch.tensor(np.array(cond_list), dtype=torch.float32)

    if x0_tensor.ndim == 3:
        x0_tensor = x0_tensor.unsqueeze(2)
    if cond_tensor.ndim == 3:
        cond_tensor = cond_tensor.unsqueeze(2)

    return x0_tensor, cond_tensor


def train(
    csv_path="outputs/petevents_bev_traffic_video_full_pet2p0.csv",
    checkpoint_dir="checkpoints",
    Th=8,
    batch_size=32,
    epochs=50,
    lr=1e-3,
    num_steps=1000,
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    x0, cond = build_training_tensors(csv_path, Th=Th)
    print("x0 shape:", tuple(x0.shape))
    print("cond shape:", tuple(cond.shape))

    dataset = TrajectoryDataset(x0, cond)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    traj_shape = tuple(x0.shape[1:])   # (Tf,2,2)
    cond_dim = cond.shape[1]

    model = TrajectoryDiffusionModel(
        traj_shape=traj_shape,
        cond_dim=cond_dim,
        num_steps=num_steps,
        device=device,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    best_path = os.path.join(checkpoint_dir, "traj_diffusion_best.pt")
    last_path = os.path.join(checkpoint_dir, "traj_diffusion_last.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        count = 0

        for batch_x0, batch_cond in loader:
            batch_x0 = batch_x0.to(device)
            batch_cond = batch_cond.to(device)

            batch_x0_flat = batch_x0.view(batch_x0.shape[0], -1)

            optimizer.zero_grad()
            loss = model(batch_x0_flat, batch_cond)
            loss.backward()
            optimizer.step()

            running += float(loss.item()) * batch_x0.shape[0]
            count += batch_x0.shape[0]

        epoch_loss = running / max(count, 1)
        print(f"Epoch {epoch:03d}/{epochs} | loss={epoch_loss:.6f}")

        torch.save({
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "traj_shape": traj_shape,
            "cond_dim": int(cond_dim),
            "num_steps": int(num_steps),
            "th": int(Th),
        }, last_path)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "traj_shape": traj_shape,
                "cond_dim": int(cond_dim),
                "num_steps": int(num_steps),
                "th": int(Th),
                "best_loss": float(best_loss),
            }, best_path)
            print("Saved best checkpoint to:", best_path)

    print("Training complete.")
    print("Best checkpoint:", best_path)
    print("Last checkpoint:", last_path)


if __name__ == "__main__":
    train()
