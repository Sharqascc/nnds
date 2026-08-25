import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.diffusion.complete_ddpm import LinearNoiseScheduler
from src.diffusion.traffic_diffusion.transformer_diffusion import (
    TransformerTrajectoryDiffusion,
)


def load_position_data(csv_path, Th=16):
    df = pd.read_csv(csv_path)
    df = df.sort_values(["event_id", "frame"])
    target_list, cond_list = [], []
    for eid, grp in df.groupby("event_id"):
        if len(grp) < Th:
            continue
        sub = grp.iloc[:Th]
        target = sub[["x_i", "y_i"]].values.astype(np.float32)
        cond = sub[["x_j", "y_j"]].values.astype(np.float32)
        target -= target[0]
        cond -= cond[0]
        target_list.append(target)
        cond_list.append(cond)
    if not target_list:
        return None
    targets = np.array(target_list)
    conds = np.array(cond_list)
    mean = targets.mean(axis=(0, 1), keepdims=True)
    std = targets.std(axis=(0, 1), keepdims=True) + 1e-6
    return (
        torch.from_numpy((targets - mean) / std).float(),
        torch.from_numpy((conds - mean) / std).float(),
        mean,
        std,
    )


def train_transformer_diffusion(
    csv_path="outputs/diffusion_del4_v4.csv",
    Th=16,
    epochs=30,
    batch_size=32,
    lr=1e-4,
    num_timesteps=200,
    checkpoint_dir="checkpoints_transformer_ddpm",
):
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    data = load_position_data(csv_path, Th=Th)
    if data is None:
        print("No data")
        return
    targets, conds, mean, std = data
    N = targets.shape[0]
    print(f"Training on {N} samples")

    model = TransformerTrajectoryDiffusion(
        traj_dim=2, cond_dim=2, hidden_dim=64, num_heads=4, num_layers=2, max_len=Th
    )
    scheduler = LinearNoiseScheduler(num_timesteps=num_timesteps)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(N)
        epoch_loss = 0.0
        for i in range(0, N, batch_size):
            idx = perm[i : i + batch_size]
            x0 = targets[idx]
            cond = conds[idx]
            t = torch.randint(0, num_timesteps, (len(idx),))
            noise = torch.randn_like(x0)
            # scheduler.add_noise expects 4D (B,T,1,2)
            x_noisy_4d = scheduler.add_noise(x0.unsqueeze(2), t, noise.unsqueeze(2))
            x_noisy = x_noisy_4d.squeeze(2)  # back to (B,T,2)
            noise_pred = model(x_noisy, cond, t)
            loss = nn.functional.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        avg_loss = epoch_loss / N
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "mean": mean,
                    "std": std,
                    "Th": Th,
                    "num_timesteps": num_timesteps,
                },
                Path(checkpoint_dir) / "transformer_ddpm_best.pt",
            )
            print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.6f} (BEST)")
        else:
            print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.6f}")
    print("Training complete.")


if __name__ == "__main__":
    train_transformer_diffusion()
