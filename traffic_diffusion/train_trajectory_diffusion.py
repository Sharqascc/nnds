import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from traffic_diffusion.trajectory_diffusion import TrajectoryDiffusionModel

def build_velocity_tensors(csv_path, Th=16, scaler_stats=None):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV not found at {csv_path}")
        
    df = pd.read_csv(csv_path)
    id_col = next((c for c in ["conflict_id", "event_id", "pair_id", "track_id", "id"] if c in df.columns), None)
    
    if id_col is None:
        df["conflict_id"] = np.arange(len(df)) // Th
        id_col = "conflict_id"

    vel_trajs = []
    cond_vecs = []

    for _, grp in df.groupby(id_col):
        if "frame" in grp.columns:
            grp = grp.sort_values("frame")
            
        if len(grp) < Th:
            continue
        sub = grp.iloc[:Th]
        
        ti = sub[["x_i", "y_i"]].values.astype(np.float32)
        tj = sub[["x_j", "y_j"]].values.astype(np.float32)
        
        v_i = np.zeros_like(ti)
        v_i[1:] = np.diff(ti, axis=0)
        
        v_j = np.zeros_like(tj)
        v_j[1:] = np.diff(tj, axis=0)
        
        cond = np.hstack([(tj[0] - ti[0]), (v_j[0] - v_i[0])])
        
        vel_trajs.append(v_i)
        cond_vecs.append(cond)

    vel_arr = np.array(vel_trajs, dtype=np.float32)   # (N, 16, 2)
    cond_arr = np.array(cond_vecs, dtype=np.float32) # (N, 4)

    if scaler_stats is not None:
        mean = scaler_stats["mean"]
        std = scaler_stats["std"]
        cond_mean = scaler_stats["cond_mean"]
        cond_std = scaler_stats["cond_std"]
    else:
        mean = vel_arr.mean(axis=(0, 1), keepdims=True)
        std = vel_arr.std(axis=(0, 1), keepdims=True) + 1e-5
        cond_mean = cond_arr.mean(axis=0, keepdims=True)
        cond_std = cond_arr.std(axis=0, keepdims=True) + 1e-5

    vel_norm = (vel_arr - mean) / std
    cond_norm = (cond_arr - cond_mean) / cond_std
    
    vel_tensor = torch.tensor(vel_norm, dtype=torch.float32).unsqueeze(2) # (N, 16, 1, 2)
    cond_tensor = torch.tensor(cond_norm, dtype=torch.float32)

    stats = {
        "mean": mean,
        "std": std,
        "cond_mean": cond_mean,
        "cond_std": cond_std
    }

    return vel_tensor, cond_tensor, stats

def train(train_csv_path="outputs/petevents_train.csv", checkpoint_dir="checkpoints", epochs=50, batch_size=32, lr=1e-3, Th=16):
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"📦 Loading training data from {train_csv_path}...")
    vel_tensor, cond_tensor, stats = build_velocity_tensors(train_csv_path, Th=Th)
    
    dataset = TensorDataset(vel_tensor, cond_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = TrajectoryDiffusionModel(traj_shape=(Th, 1, 2), cond_dim=4, hidden_dim=128).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_loss = float("inf")
    ckpt_path = os.path.join(checkpoint_dir, "traj_diffusion_best.pt")
    
    print(f"🔥 Training Flow Matching Velocity Model for {epochs} epochs on {device}...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        for batch_vel, batch_cond in loader:
            batch_vel = batch_vel.to(device)
            batch_cond = batch_cond.to(device)
            
            optimizer.zero_grad()
            loss = model.compute_loss(batch_vel, batch_cond)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item() * len(batch_vel)
            
        scheduler.step()
        avg_loss = epoch_loss / len(dataset)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/{epochs}] - Flow Matching Loss: {avg_loss:.6f}")
            
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "state_dict": model.state_dict(),
                "stats": stats,
                "model_type": "velocity_diffusion"
            }, ckpt_path)

    print(f"✅ Training complete. Checkpoint saved to {ckpt_path} (Best Loss: {best_loss:.6f})")

if __name__ == "__main__":
    train()
