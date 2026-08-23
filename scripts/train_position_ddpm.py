import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from src.diffusion.complete_ddpm import LinearNoiseScheduler, TrajectoryUNet1D

def load_position_data(csv_path, Th=16):
    df = pd.read_csv(csv_path)
    df = df.sort_values(["event_id", "frame"])
    target_list = []
    cond_list = []
    for eid, grp in df.groupby("event_id"):
        if len(grp) < Th:
            continue
        sub = grp.iloc[:Th]
        # target: x_i,y_i
        target = sub[["x_i", "y_i"]].values.astype(np.float32)   # (Th,2)
        cond = sub[["x_j", "y_j"]].values.astype(np.float32)     # (Th,2)
        # center target and cond at their first frame
        target -= target[0]
        cond -= cond[0]
        target_list.append(target)
        cond_list.append(cond)
    if not target_list:
        return None
    targets = np.array(target_list)    # (N,Th,2)
    conds = np.array(cond_list)
    # Normalize targets and conditions globally
    mean = targets.mean(axis=(0,1), keepdims=True)
    std = targets.std(axis=(0,1), keepdims=True) + 1e-6
    targets_norm = (targets - mean) / std
    conds_norm = (conds - mean) / std
    # Reshape to (N, Th, 1, 2) for model
    targets_norm = targets_norm[:, :, None, :]
    conds_norm = conds_norm[:, :, None, :]
    return torch.from_numpy(targets_norm).float(), torch.from_numpy(conds_norm).float(), mean, std, target_list, cond_list

def train_position_ddpm(csv_path="outputs/diffusion_del4_v4.csv", Th=16, epochs=50, batch_size=32, lr=1e-4, num_timesteps=200, checkpoint_dir="checkpoints_ddpm_pos"):
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    data = load_position_data(csv_path, Th=Th)
    if data is None:
        print("No data loaded")
        return
    targets, conds, mean, std, _, _ = data
    N = targets.shape[0]
    print(f"Training on {N} samples")

    input_dim = Th * 2  # model input_dim = T*2
    model = TrajectoryUNet1D(input_dim=input_dim, cond_dim=input_dim, hidden_dim=128, num_layers=3)
    scheduler = LinearNoiseScheduler(num_timesteps=num_timesteps)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')

    for epoch in range(1, epochs+1):
        model.train()
        perm = torch.randperm(N)
        epoch_loss = 0.0
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            x0 = targets[idx]
            cond = conds[idx]
            t = torch.randint(0, num_timesteps, (len(idx),))
            noise = torch.randn_like(x0)
            x_noisy = scheduler.add_noise(x0, t, noise)
            noise_pred = model(x_noisy, t, cond)
            loss = nn.functional.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        avg_loss = epoch_loss / N
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "mean": mean,
                "std": std,
                "Th": Th,
                "num_timesteps": num_timesteps,
            }, Path(checkpoint_dir) / "position_ddpm_best.pt")
            print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.6f} (BEST)")
        else:
            print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.6f}")
    print(f"Training complete. Best loss: {best_loss:.6f}")

if __name__ == "__main__":
    train_position_ddpm(csv_path="outputs/diffusion_del4_v4.csv", Th=16, epochs=50)
