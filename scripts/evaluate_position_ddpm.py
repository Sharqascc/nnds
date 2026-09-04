import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from scipy.stats import wasserstein_distance

from src.diffusion.complete_ddpm import LinearNoiseScheduler, TrajectoryUNet1D


def load_position_data_subset(csv_path, Th=16, max_events=200):
    df = pd.read_csv(csv_path)
    df = df.sort_values(["event_id", "frame"])
    event_ids = df["event_id"].unique()[:max_events]
    df = df[df["event_id"].isin(event_ids)]
    target_list = []
    cond_list = []
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
    return np.array(target_list), np.array(cond_list)


def evaluate_position_ddpm(
    csv_path="outputs/diffusion_del4_v4.csv",
    checkpoint_path="checkpoints_ddpm_pos/position_ddpm_best.pt",
    Th=16,
    max_events=200,
    K=10,
    num_steps=50,
    dt=0.1,
):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    targets, conds = load_position_data_subset(csv_path, Th=Th, max_events=max_events)
    mean = ckpt["mean"]
    std = ckpt["std"]
    targets_norm = (targets - mean) / std
    conds_norm = (conds - mean) / std
    targets_tensor = torch.from_numpy(targets_norm[:, :, None, :]).float()
    cond_tensor = torch.from_numpy(conds_norm[:, :, None, :]).float()

    input_dim = Th * 2
    model = TrajectoryUNet1D(input_dim=input_dim, cond_dim=input_dim, hidden_dim=128, num_layers=3)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    scheduler = LinearNoiseScheduler(num_timesteps=ckpt["num_timesteps"])

    N = targets_tensor.shape[0]
    all_gen = []
    with torch.no_grad():
        for _ in range(K):
            x = torch.randn(N, Th, 1, 2)
            for t in torch.linspace(ckpt["num_timesteps"] - 1, 0, num_steps).long():
                t_tensor = torch.full((N,), t, dtype=torch.long)
                noise_pred = model(x, t_tensor, cond_tensor)
                x = scheduler.sample_prev_timestep(x, t_tensor, noise_pred)
            all_gen.append(x.numpy().squeeze(2))

    all_gen = np.stack(all_gen, axis=1)  # (N, K, Th, 2)
    # Unnormalize
    all_gen = all_gen * std + mean

    # Select best of K by minADE
    errors = np.linalg.norm(all_gen - targets[:, None, :, :], axis=-1)
    best_k = np.argmin(np.mean(errors, axis=-1), axis=1)
    best = all_gen[np.arange(N), best_k]

    # Metrics
    ade = np.mean(np.linalg.norm(best - targets, axis=-1))
    fde = np.mean(np.linalg.norm(best[:, -1] - targets[:, -1], axis=-1))

    vel = np.diff(best, axis=1) / dt
    acc = np.diff(vel, axis=1) / dt
    jerk = np.diff(acc, axis=1) / dt
    acc_viol = np.mean(np.linalg.norm(acc, axis=-1) > 4.5) * 100.0
    jerk_viol = np.mean(np.linalg.norm(jerk, axis=-1) > 2.5) * 100.0

    # PET: use condition trajectories (unnormalized)
    gen_pets = []
    for b in range(N):
        dist_matrix = np.linalg.norm(best[b][:, None, :] - conds[b][None, :, :], axis=-1)
        min_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        gen_pets.append(abs(min_idx[0] - min_idx[1]) * dt)

    # Real PET values from CSV (first pet per event)
    df = pd.read_csv(csv_path)
    event_ids = df["event_id"].unique()[:max_events]
    real_pets = []
    for eid in event_ids:
        sub = df[df["event_id"] == eid]
        if len(sub) > 0 and "pet" in sub.columns:
            real_pets.append(float(sub["pet"].iloc[0]))
        else:
            real_pets.append(0.5)
    real_pets = np.array(real_pets[:N])

    w1 = wasserstein_distance(real_pets, np.array(gen_pets)) if len(real_pets) > 0 else 0.0

    print("=" * 60)
    print("POSITION DDPM EVALUATION")
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
    evaluate_position_ddpm()
