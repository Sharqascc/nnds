import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from traffic_diffusion.trajectory_diffusion import TrajectoryDiffusionModel

def build_normalized_tensors(csv_path, Th=16, scaler_stats=None):
    df = pd.read_csv(csv_path)
    det_path = Path(csv_path).parent / "petevents_bev_traffic_video_full_pet2p0_detections.csv"
    if not det_path.exists():
        det_path = Path("outputs/petevents_bev_traffic_video_full_pet2p0_detections.csv")
    
    det_df = pd.read_csv(det_path)

    cand_i = ["track_a", "track_id_i", "actor_i", "track_i", "id_i", "i", "actor1", "track1"]
    cand_j = ["track_b", "track_id_j", "actor_j", "track_j", "id_j", "j", "actor2", "track2"]

    id_col_i = next((c for c in cand_i if c in df.columns), None)
    id_col_j = next((c for c in cand_j if c in df.columns), None)

    if id_col_i is None or id_col_j is None:
        raise KeyError(f"Could not find actor columns in {df.columns.tolist()}")

    track_col = next((c for c in ["track_id", "actor_id", "id", "track"] if c in det_df.columns), "track_id")
    frame_col = next((c for c in ["frame", "frame_idx", "t"] if c in det_df.columns), "frame")
    
    x_col = next((c for c in ["world_x", "cx", "x"] if c in det_df.columns), "x")
    y_col = next((c for c in ["world_y", "cy", "y"] if c in det_df.columns), "y")

    x0_list, cond_list, real_pets = [], [], []

    for _, row in df.iterrows():
        id_i, id_j = row[id_col_i], row[id_col_j]
        ti = det_df[det_df[track_col] == id_i][[frame_col, x_col, y_col]].sort_values(frame_col)[[x_col, y_col]].values
        tj = det_df[det_df[track_col] == id_j][[frame_col, x_col, y_col]].sort_values(frame_col)[[x_col, y_col]].values

        if len(ti) > 0 and len(tj) > 0:
            if len(ti) < Th: ti = np.pad(ti, ((0, Th - len(ti)), (0, 0)), mode="edge")
            if len(tj) < Th: tj = np.pad(tj, ((0, Th - len(tj)), (0, 0)), mode="edge")

            x0_list.append(ti[:Th])
            cond_list.append(tj[:Th])
            if "pet" in row: real_pets.append(row["pet"])

    x0_arr = np.array(x0_list, dtype=np.float32)
    cond_arr = np.array(cond_list, dtype=np.float32)

    # Scale pixel coordinates to meter-scale (~0.05 conversion factor) if world coordinates aren't present
    if np.max(np.abs(x0_arr)) > 50.0:
        x0_arr = x0_arr * 0.05
        cond_arr = cond_arr * 0.05

    if scaler_stats is None:
        combined = np.concatenate([x0_arr, cond_arr], axis=0)
        mean = combined.mean(axis=(0, 1), keepdims=True)
        std = combined.std(axis=(0, 1), keepdims=True) + 1e-6
    else:
        mean, std = scaler_stats["mean"], scaler_stats["std"]

    x0_norm = (x0_arr - mean) / std
    cond_norm = (cond_arr - mean) / std

    x0_tensor = torch.tensor(x0_norm, dtype=torch.float32).unsqueeze(2)     # (B, Th, 1, 2)
    cond_tensor = torch.tensor(cond_norm, dtype=torch.float32).unsqueeze(2)   # (B, Th, 1, 2)

    stats = {"mean": mean, "std": std}
    return x0_tensor, cond_tensor, stats, x0_arr, cond_arr, np.array(real_pets)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print(f"🚀 TRAINING SCALED DIFFUSION MODEL ON {device}")
    print("=" * 65)

    train_csv = "outputs/petevents_train.csv"
    val_csv = "outputs/petevents_val.csv"
    Th = 16
    epochs = 100
    lr = 1e-3

    x0_train, cond_train, stats, _, _, _ = build_normalized_tensors(train_csv, Th=Th)
    x0_val, cond_val, _, _, _, _ = build_normalized_tensors(val_csv, Th=Th, scaler_stats=stats)

    x0_train, cond_train = x0_train.to(device), cond_train.to(device)
    x0_val, cond_val = x0_val.to(device), cond_val.to(device)

    print(f"📊 Training Samples: {len(x0_train)} | Val Samples: {len(x0_val)} | Input Shape: {tuple(x0_train.shape[1:])}")

    model = TrajectoryDiffusionModel(traj_shape=(Th, 1, 2), cond_dim=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    os.makedirs("checkpoints", exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        loss = model(x0_train, cond_train)
        
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = model(x0_val, cond_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "traj_shape": (Th, 1, 2),
                "cond_dim": 2,
                "best_val_loss": best_val_loss,
                "mean": stats["mean"],
                "std": stats["std"]
            }
            torch.save(checkpoint, "checkpoints/traj_diffusion_best.pt")

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} {'🔥 Best' if val_loss == best_val_loss else ''}")

    print("=" * 65)
    print(f"✅ Training completed! Best Checkpoint Val Loss: {best_val_loss:.4f}")
    print("=" * 65)

if __name__ == "__main__":
    train()
