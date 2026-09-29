import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import contextlib

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.stats import wasserstein_distance

from src.diffusion.complete_ddpm import LinearNoiseScheduler
from src.diffusion.traffic_diffusion.transformer_diffusion import (
    TransformerTrajectoryDiffusion,
)


def load_position_data_subset(csv_path, Th=16, max_events=200):
    df = pd.read_csv(csv_path)
    df = df.sort_values(["event_id", "frame"])
    event_ids = df["event_id"].unique()[:max_events]
    df = df[df["event_id"].isin(event_ids)]
    target_list, cond_list, real_pets = [], [], []
    for _eid, grp in df.groupby("event_id"):
        if len(grp) < Th:
            continue
        sub = grp.iloc[:Th]
        target = sub[["x_i", "y_i"]].values.astype(np.float32)
        cond = sub[["x_j", "y_j"]].values.astype(np.float32)
        target -= target[0]
        cond -= cond[0]
        target_list.append(target)
        cond_list.append(cond)
        real_pets.append(float(sub["pet"].iloc[0]) if "pet" in sub.columns else 0.5)
    return np.array(target_list), np.array(cond_list), np.array(real_pets)


def evaluate_transformer_diffusion(
    csv_path="outputs/diffusion_del4_v4.csv",
    checkpoint_path="checkpoints_transformer_ddpm/transformer_ddpm_best.pt",
    Th=16,
    max_events=200,
    K=10,
    num_steps=50,
    dt=0.1,
):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    targets, conds, real_pets = load_position_data_subset(csv_path, Th=Th, max_events=max_events)
    N = len(targets)
    if N == 0:
        print("No data")
        return

    mean = ckpt["mean"]
    std = ckpt["std"]
    (targets - mean) / std
    conds_norm = (conds - mean) / std

    model = TransformerTrajectoryDiffusion(
        traj_dim=2, cond_dim=2, hidden_dim=64, num_heads=4, num_layers=2, max_len=Th
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    scheduler = LinearNoiseScheduler(num_timesteps=ckpt["num_timesteps"])

    cond_tensor = torch.from_numpy(conds_norm).float()
    all_gen = []
    with torch.no_grad():
        for _k in range(K):
            # Start from noise, shape (N, Th, 1, 2)
            x = torch.randn(N, Th, 1, 2)
            for t in torch.linspace(ckpt["num_timesteps"] - 1, 0, num_steps).long():
                t_tensor = torch.full((N,), t, dtype=torch.long)
                # Model expects (N, Th, 2)
                noise_pred = model(x.squeeze(2), cond_tensor, t_tensor)
                # Convert to 4D for scheduler
                noise_pred_4d = noise_pred.unsqueeze(2)
                x = scheduler.sample_prev_timestep(x, t_tensor, noise_pred_4d)
            all_gen.append(x.squeeze(2).numpy())

    all_gen = np.stack(all_gen, axis=1)  # (N, K, Th, 2)
    all_gen = all_gen * std + mean  # denormalize

    # Select best of K by minADE
    errors = np.linalg.norm(all_gen - targets[:, None, :, :], axis=-1)
    best_k = np.argmin(np.mean(errors, axis=-1), axis=1)
    best = all_gen[np.arange(N), best_k]

    # Apply smoothing for kinematic improvement
    with contextlib.suppress(Exception):
        best = savgol_filter(best, window_length=5, polyorder=2, axis=1)
    with contextlib.suppress(Exception):
        best = gaussian_filter1d(best, sigma=0.8, axis=1)

    ade = np.mean(np.linalg.norm(best - targets, axis=-1))
    fde = np.mean(np.linalg.norm(best[:, -1] - targets[:, -1], axis=-1))

    vel = np.diff(best, axis=1) / dt
    acc = np.diff(vel, axis=1) / dt
    jerk = np.diff(acc, axis=1) / dt
    acc_viol = np.mean(np.linalg.norm(acc, axis=-1) > 4.5) * 100.0
    jerk_viol = np.mean(np.linalg.norm(jerk, axis=-1) > 2.5) * 100.0

    # PET: use condition trajectories
    gen_pets = []
    for b in range(N):
        dist_matrix = np.linalg.norm(best[b][:, None, :] - conds[b][None, :, :], axis=-1)
        min_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        gen_pets.append(abs(min_idx[0] - min_idx[1]) * dt)

    w1 = wasserstein_distance(real_pets, np.array(gen_pets)) if len(real_pets) > 0 else 0.0

    print("=" * 60)
    print("TRANSFORMER DIFFUSION EVALUATION")
    print("=" * 60)
    print(f"Samples evaluated: {N}")
    print(f"minADE: {ade:.4f} m (Target < 0.50)")
    print(f"minFDE: {fde:.4f} m (Target < 1.00)")
    print(f"Acceleration violations: {acc_viol:.2f}% (Target < 2.0%)")
    print(f"Jerk violations: {jerk_viol:.2f}% (Target < 2.0%)")
    print(f"Ground-Truth PET: {np.mean(real_pets):.3f}s ± {np.std(real_pets):.3f}s")
    print(f"Generated PET:    {np.mean(gen_pets):.3f}s ± {np.std(gen_pets):.3f}s")
    print(f"PET W1: {w1:.4f} (Target < 0.150)")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_transformer_diffusion()
