import math

import numpy as np
import pandas as pd
import torch
from scipy.signal import savgol_filter
from scipy.stats import wasserstein_distance
from torch import nn

# ============ DDPM Components ============


class LinearNoiseScheduler:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_timesteps) ** 2
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)

    def add_noise(self, x, t, noise):
        sqrt_alpha_bar = self.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1).to(x.device)
        sqrt_one_minus_alpha_bar = (
            self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1).to(x.device)
        )
        return sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise

    def sample_prev_timestep(self, x, t, noise_pred):
        alpha = self.alphas[t[0]].item()
        alpha_bar = self.alpha_cumprod[t[0]].item()
        alpha_bar_prev = self.alpha_cumprod[t[0] - 1].item() if t[0] > 0 else 1.0

        pred_x0 = (x - math.sqrt(1 - alpha_bar) * noise_pred) / math.sqrt(alpha_bar)
        pred_x0 = pred_x0.clamp(-1, 1)

        sigma = math.sqrt(1 - alpha_bar_prev) * math.sqrt(1 - alpha) / math.sqrt(1 - alpha_bar)
        direction = math.sqrt(1 - alpha_bar_prev) * noise_pred if t[0] > 0 else 0

        x_prev = math.sqrt(alpha_bar_prev) * pred_x0 + direction

        if t[0] > 0:
            x_prev = x_prev + sigma * torch.randn_like(x)

        return x_prev


# ============ 1D CNN UNet ============


class TrajectoryUNet1D(nn.Module):
    def __init__(self, input_dim=32, cond_dim=32, hidden_dim=128, num_layers=3):
        super().__init__()
        self.T = input_dim // 2

        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.cond_embed = nn.Linear(cond_dim, hidden_dim)

        # Simple 1D CNN
        self.conv1 = nn.Conv1d(4, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
        self.norm = nn.GroupNorm(1, hidden_dim)
        self.output = nn.Conv1d(hidden_dim, 2, 1)
        self.act = nn.SiLU()

    def forward(self, x, t, cond):
        batch = x.shape[0]
        self.time_embed(t.view(-1, 1).float())
        self.cond_embed(cond.view(batch, -1))

        # Concatenate x and cond along feature dim
        x_cat = torch.cat([x, cond], dim=2)  # (batch, T, 2, 2)
        x_cat = x_cat.permute(0, 2, 1, 3).reshape(batch, 4, self.T)  # (batch, 4, T)

        # CNN
        x_cat = self.act(self.conv1(x_cat))
        x_cat = self.act(self.conv2(x_cat))
        x_cat = self.act(self.conv3(x_cat))
        x_cat = self.norm(x_cat)

        out = self.output(x_cat)  # (batch, 2, T)
        out = out.transpose(1, 2).unsqueeze(2)  # (batch, T, 1, 2)

        return out

    @torch.no_grad()
    def sample(self, cond, scheduler, num_steps=50):
        device = cond.device
        batch, T = cond.shape[0], cond.shape[1]
        x = torch.randn(batch, T, 1, 2, device=device)

        for t in torch.linspace(scheduler.num_timesteps - 1, 0, num_steps).long():
            t_tensor = torch.full((batch,), t, device=device, dtype=torch.long)
            noise_pred = self.forward(x, t_tensor, cond)
            x = scheduler.sample_prev_timestep(x, t_tensor, noise_pred)

        return x


# ============ Data Loading ============


def parse_traj(traj_str, Th=16):
    """Parse trajectory string like [(frame, x, y), ...]"""
    import ast

    try:
        traj_list = ast.literal_eval(traj_str)
        xy_data = [(t[1], t[2]) for t in traj_list[:Th]]
        return np.array(xy_data, dtype=np.float32) if len(xy_data) >= Th else None
    except:
        return None


def load_data_from_csv(csv_path, Th=16):
    df = pd.read_csv(csv_path)

    # Check if we have the new format (world_traj_i, world_traj_j)
    if "world_traj_i" in df.columns and "world_traj_j" in df.columns:
        gt_trajs, cond_trajs, start_pos_arr = [], [], []

        for idx, row in df.iterrows():
            traj_i = parse_traj(row["world_traj_i"], Th=Th)
            traj_j = parse_traj(row["world_traj_j"], Th=Th)

            if traj_i is not None and traj_j is not None:
                gt_trajs.append(traj_i)
                cond_trajs.append(traj_j)
                start_pos_arr.append(traj_i[0])

        if len(gt_trajs) == 0:
            return None, None, None

        return np.array(gt_trajs), np.array(cond_trajs), np.array(start_pos_arr)

    # Old format
    id_col = next(
        (c for c in ["conflict_id", "event_id", "pair_id", "track_id", "id"] if c in df.columns),
        None,
    )

    gt_trajs, cond_trajs, start_pos_arr = [], [], []

    for _, grp in df.groupby(id_col):
        if "frame" in grp.columns:
            grp = grp.sort_values("frame")
        if len(grp) < Th:
            continue

        sub = grp.iloc[:Th]
        gt_trajs.append(sub[["x_i", "y_i"]].values.astype(np.float32))
        cond_trajs.append(sub[["x_j", "y_j"]].values.astype(np.float32))
        start_pos_arr.append(sub[["x_i", "y_i"]].values[0])

    if len(gt_trajs) == 0:
        return None, None, None

    return np.array(gt_trajs), np.array(cond_trajs), np.array(start_pos_arr)


def build_velocity_tensors(csv_path, Th=16):
    result = load_data_from_csv(csv_path, Th=Th)
    if result[0] is None:
        return None

    gt_trajs, cond_trajs, start_pos_arr = result
    mu = np.mean(gt_trajs, axis=(0, 1, 2), keepdims=True)
    sigma = np.std(gt_trajs, axis=(0, 1, 2), keepdims=True) + 1e-8

    vel_tensor_norm = torch.from_numpy((gt_trajs - mu) / sigma).unsqueeze(2).float()
    cond_tensor = torch.from_numpy((cond_trajs - mu) / sigma).unsqueeze(2).float()

    return (
        vel_tensor_norm,
        cond_tensor,
        mu,
        sigma,
        gt_trajs,
        torch.from_numpy(start_pos_arr).float(),
    )


# ============ Training ============


def train_model(
    train_csv,
    input_dim=32,
    hidden_dim=128,
    epochs=100,
    batch_size=16,
    lr=1e-4,
    save_path="checkpoints/best.pt",
    num_timesteps=1000,
):
    result = build_velocity_tensors(train_csv, Th=input_dim // 2)
    if result is None:
        print("Error loading data")
        return

    vel_tensor_norm, cond_tensor, mu, sigma, _, _ = result
    N = vel_tensor_norm.shape[0]
    print(f"Training with {N} samples")

    model = TrajectoryUNet1D(input_dim=input_dim, cond_dim=input_dim, hidden_dim=hidden_dim)
    scheduler = LinearNoiseScheduler(num_timesteps=num_timesteps)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(N)
        epoch_loss = 0.0

        for i in range(0, N, batch_size):
            idx = perm[i : i + batch_size]
            x = vel_tensor_norm[idx]
            cond = cond_tensor[idx]

            t = torch.randint(0, num_timesteps, (len(idx),))
            noise = torch.randn_like(x)
            x_noisy = scheduler.add_noise(x, t, noise)

            noise_pred = model(x_noisy, t, cond)
            loss = nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(1, N // batch_size)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "mu": mu,
                    "sigma": sigma,
                    "num_timesteps": num_timesteps,
                },
                save_path,
            )
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} (BEST)")
        else:
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    print(f"Training complete. Best loss: {best_loss:.4f}")


# ============ Evaluation ============


def evaluate_model(test_csv, checkpoint_path, K=10, Th=16, dt=0.1):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # Get Th from checkpoint if available
    if "traj_shape" in checkpoint:
        Th = checkpoint["traj_shape"][0]
    elif "input_dim" in checkpoint:
        Th = checkpoint["input_dim"] // 2
    result = load_data_from_csv(test_csv, Th=Th)
    gt_trajs, cond_trajs, start_pos_arr = result[:3]
    real_pets = None if len(result) < 4 else result[3]

    result = build_velocity_tensors(test_csv, Th=Th)
    if result is None:
        print("Error loading test data")
        return

    vel_tensor_norm, cond_tensor, mu, sigma, _, _ = result
    N, T = vel_tensor_norm.shape[0], vel_tensor_norm.shape[1]
    print(f"Test samples: {N}, horizon T: {T}")

    input_dim = checkpoint.get("input_dim", checkpoint["traj_shape"][0] * 2)
    model = TrajectoryUNet1D(input_dim=input_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scheduler = LinearNoiseScheduler(num_timesteps=checkpoint["num_timesteps"])

    all_gen = []
    with torch.no_grad():
        for _ in range(K):
            gen = model.sample(cond_tensor, scheduler, num_steps=50).numpy()
            all_gen.append(gen)

    all_gen = np.stack(all_gen, axis=0)
    all_gen = all_gen * sigma + mu

    # Integrate to positions
    all_gen_pos = np.zeros((N, K, T, 2))
    for k in range(K):
        all_gen_pos[:, k, 0] = start_pos_arr
        all_gen_pos[:, k, 1:] = (
            start_pos_arr[:, None] + np.cumsum(all_gen[k, :, :-1, :], axis=1) * dt
        )

    all_gen_pos = savgol_filter(all_gen_pos, window_length=5, polyorder=3, axis=2)

    # Compute metrics
    min_ade, min_fde = [], []
    for b in range(N):
        ade = np.mean(np.abs(all_gen_pos[b] - gt_trajs[b]), axis=(1, 2))
        fde = np.abs(all_gen_pos[b, :, -1] - gt_trajs[b, -1]).mean(axis=1)
        min_ade.append(ade.min())
        min_fde.append(fde.min())

    print(f"minADE: {np.mean(min_ade):.2f}m (Target: <0.50m)")
    print(f"minFDE: {np.mean(min_fde):.2f}m (Target: <1.00m)")

    # Kinematics
    vel = np.diff(all_gen_pos[:, 0], axis=1) / dt
    acc = np.diff(vel, axis=1) / dt
    acc_violations = (np.abs(acc) > 4.5).mean() * 100
    print(f"Acceleration violations: {acc_violations:.2f}% (Target: <2.0%)")

    # PET
    gen_pets = []
    for b in range(N):
        dist = np.linalg.norm(all_gen_pos[b, 0, :, None] - cond_trajs[b, None, :], axis=-1)
        min_idx = np.unravel_index(np.argmin(dist), dist.shape)
        gen_pets.append(abs(min_idx[0] - min_idx[1]) * dt)

    w1 = wasserstein_distance(real_pets, gen_pets)
    print(f"PET W1: {w1:.4f} (Target: <0.150)")
    print("Evaluation complete!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_model("outputs/petevents_train.csv", input_dim=18)
    else:
        evaluate_model("outputs/petevents_test.csv", "checkpoints/best.pt")
