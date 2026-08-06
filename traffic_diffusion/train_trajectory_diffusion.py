import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from traffic_diffusion.trajectory_diffusion import TrajectoryDiffusionModel

def build_normalized_tensors(csv_path, Th=16, scaler_stats=None, augment=False):
    """
    Extracts trajectory pairs from CSV and converts them into relative displacements (p_t - p_0).
    Flexibly detects event ID columns (event_id, conflict_id, pair_id, track_id, id).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV not found at {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    # 1. Flexible Event ID Detection
    id_candidates = ["event_id", "conflict_id", "pair_id", "track_id", "case_id", "id", "event"]
    id_col = next((c for c in id_candidates if c in df.columns), None)
    
    if id_col is None:
        print(f"⚠️ No explicit event ID column found in {csv_path}. Auto-chunking into sequences of length {Th}.")
        df["event_id"] = np.arange(len(df)) // Th
        id_col = "event_id"
    else:
        print(f"ℹ️ Found event identifier column: '{id_col}'")

    # 2. Check Coordinate Columns
    required_coords = ["x_i", "y_i", "x_j", "y_j"]
    for col in required_coords:
        if col not in df.columns:
            raise ValueError(f"Missing required coordinate column '{col}' in {csv_path}. Available columns: {list(df.columns)}")
            
    rel_list, cond_list, start_pos_list, real_pets = [], [], [], []
    
    for event_id, group in df.groupby(id_col):
        if "frame" in group.columns:
            group = group.sort_values("frame")
        
        if len(group) < Th:
            continue
            
        ti_seq = group[["x_i", "y_i"]].values[:Th]
        tj_seq = group[["x_j", "y_j"]].values[:Th]
        
        # Reference position: starting position of vehicle i (p_0)
        start_pos = ti_seq[0].copy()
        
        # Calculate relative displacements (p_t - p_0)
        ti_rel = ti_seq - start_pos
        tj_rel = tj_seq[0] - start_pos # Condition on initial position of vehicle j
        
        rel_list.append(ti_rel)
        cond_list.append(tj_rel)
        start_pos_list.append(start_pos)
        
        if "pet" in group.columns:
            real_pets.append(group["pet"].iloc[0])
        elif "PET" in group.columns:
            real_pets.append(group["PET"].iloc[0])
        else:
            real_pets.append(1.5)
            
    if len(rel_list) == 0:
        raise ValueError(f"No valid trajectory sequences of length {Th} found in {csv_path}")
        
    x0_arr = np.array(rel_list, dtype=np.float32)       # Shape: (N, Th, 2)
    cond_arr = np.array(cond_list, dtype=np.float32)     # Shape: (N, 2)
    start_pos_arr = np.array(start_pos_list, dtype=np.float32) # Shape: (N, 2)
    real_pets = np.array(real_pets, dtype=np.float32)
    
    # Data Augmentation (Random Rotation during Training)
    if augment:
        angles = np.random.uniform(0, 2 * np.pi, size=len(x0_arr))
        cos_a, sin_a = np.cos(angles), np.sin(angles)
        R = np.stack([np.stack([cos_a, -sin_a], axis=-1), 
                      np.stack([sin_a, cos_a], axis=-1)], axis=-2) # (N, 2, 2)
        
        x0_arr = np.einsum("nti,nij->ntj", x0_arr, R)
        cond_arr = np.einsum("ni,nij->nj", cond_arr, R)

    # Dataset normalization statistics
    if scaler_stats is None:
        mean = float(np.mean(x0_arr))
        std = float(np.std(x0_arr)) + 1e-6
        stats = {"mean": mean, "std": std}
    else:
        stats = scaler_stats
        mean, std = stats["mean"], stats["std"]
        
    x0_norm = (x0_arr - mean) / std
    cond_norm = (cond_arr - mean) / std
    
    # Expand channel dimension for Temporal U-Net: (N, Th, 1, 2)
    x0_tensor = torch.tensor(x0_norm, dtype=torch.float32).unsqueeze(2)
    cond_tensor = torch.tensor(cond_norm, dtype=torch.float32)
    
    return x0_tensor, cond_tensor, stats, x0_arr, cond_arr, start_pos_arr, real_pets

def train(train_csv_path, checkpoint_dir="checkpoints", epochs=40, batch_size=32, lr=1e-3, Th=16):
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"📦 Loading training data from {train_csv_path}...")
    x0_train, cond_train, stats, _, _, _, _ = build_normalized_tensors(
        train_csv_path, Th=Th, augment=True
    )
    
    dataset = torch.utils.data.TensorDataset(x0_train, cond_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = TrajectoryDiffusionModel(traj_shape=(Th, 1, 2), cond_dim=2, hidden_dim=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_loss = float("inf")
    checkpoint_path = os.path.join(checkpoint_dir, "traj_diffusion_best.pt")
    
    print(f"🔥 Starting training for {epochs} epochs on device: {device}...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        for x_batch, c_batch in loader:
            x_batch, c_batch = x_batch.to(device), c_batch.to(device)
            optimizer.zero_grad()
            
            loss = model.p_losses(x_batch, c_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item() * len(x_batch)
            
        epoch_loss = total_loss / len(dataset)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - MSE Loss: {epoch_loss:.6f}")
            
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                "state_dict": model.state_dict(),
                "mean": stats["mean"],
                "std": stats["std"]
            }, checkpoint_path)
            
    print(f"✅ Training complete. Best model saved to {checkpoint_path} (Loss: {best_loss:.6f})")

if __name__ == "__main__":
    train("outputs/petevents_train.csv")
