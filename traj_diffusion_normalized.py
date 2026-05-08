import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# =========================================================
# REQUIRED PRE-DEFINED OBJECTS IN YOUR RUNTIME
# =========================================================
# - inputs_np: (N, Th, 4) past trajectories (x_i,y_i,x_j,y_j)
# - targets_np: (N, Tf, 4) future trajectories
# - model: diffusion UNet (or similar)
# - device: torch.device("cuda") or "cpu"
# - T_steps, betas, alphas, alphas_cumprod
# - sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod
# - sample_timesteps: function(B) -> LongTensor (B,)
# - optimizer: optimizer for model parameters
# ---------------------------------------------------------

# -----------------------------
# 1) Normalize data
# -----------------------------
if __name__ == "__main__":
    centers = inputs_np[:, -1, 0:2]  # (N, 2), last past point of agent i

    inputs_norm = inputs_np.copy()
    targets_norm = targets_np.copy()

    for k in range(inputs_np.shape[0]):
        cx, cy = centers[k]

        # past
        inputs_norm[k, :, 0] -= cx
        inputs_norm[k, :, 1] -= cy
        inputs_norm[k, :, 2] -= cx
        inputs_norm[k, :, 3] -= cy

        # future
        targets_norm[k, :, 0] -= cx
        targets_norm[k, :, 1] -= cy
        targets_norm[k, :, 2] -= cx
        targets_norm[k, :, 3] -= cy

    scale = 1.0
    inputs_norm /= scale
    targets_norm /= scale

    print("inputs_norm range:", inputs_norm.min(), inputs_norm.max())
    print("targets_norm range:", targets_norm.min(), targets_norm.max())

    # -----------------------------
    # 2) Dataset / DataLoader
    # -----------------------------
    class TrajDiffusionDatasetNorm(Dataset):
        def __init__(self, inputs_norm, targets_norm, centers):
            self.inputs = torch.from_numpy(inputs_norm).float()
            self.targets = torch.from_numpy(targets_norm).float()
            self.centers = torch.from_numpy(centers).float()

        def __len__(self):
            return self.inputs.shape[0]

        def __getitem__(self, idx):
            return {
                "idx": idx,
                "past": self.inputs[idx],    # (Th, 4), normalized
                "future": self.targets[idx], # (Tf, 4), normalized
                "center": self.centers[idx], # (2,)
            }

    dataset = TrajDiffusionDatasetNorm(inputs_norm, targets_norm, centers)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)


    Tf = targets_norm.shape[1]

    # -----------------------------
    # 3) Training loop (MSE on noise)
    # -----------------------------
    def train_diffusion(model, loader, optimizer, epochs=20):
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0

            for batch in loader:
                past = batch["past"].to(device)      # (B, Th, 4), normalized
                future = batch["future"].to(device)  # (B, Tf, 4), normalized

                B = future.shape[0]
                t = sample_timesteps(B)

                noise = torch.randn_like(future)
                alpha_t = sqrt_alphas_cumprod[t].view(B, 1, 1)
                one_minus_alpha_t = sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1)

                x_t = alpha_t * future + one_minus_alpha_t * noise
                pred_noise = model(x_t, t, cond=past)

                loss = torch.mean((pred_noise - noise) ** 2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * B

            epoch_loss /= len(dataset)
            print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.6f}")

    # -----------------------------
    # 4) Sampling (normalized space)
    # -----------------------------
    @torch.no_grad()
    def sample_future_denorm(batch, max_B=4):
        """
        Returns: (B, Tf, 4) in normalized coordinates (only scale undone).
        """
        past = batch["past"][:max_B].to(device)
        B = past.shape[0]

        x_t = torch.randn(B, Tf, 4, device=device)

        for t_step in reversed(range(T_steps)):
            t = torch.full((B,), t_step, device=device, dtype=torch.long)
            eps_theta = model(x_t, t, cond=past)

            alpha_t = alphas[t_step]
            beta_t = betas[t_step]
            alpha_bar_t = alphas_cumprod[t_step]
            alpha_bar_prev = (
                alphas_cumprod[t_step - 1]
                if t_step > 0 else torch.tensor(1.0, device=device)
            )

            posterior_variance = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
            posterior_variance = torch.clamp(posterior_variance, min=1e-20)

            x0_pred = (x_t - torch.sqrt(1.0 - alpha_bar_t) * eps_theta) / torch.sqrt(alpha_bar_t)
            x0_pred = torch.clamp(x0_pred, -1e6, 1e6)

            coef1 = torch.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar_t)
            coef2 = torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
            mean = coef1 * x0_pred + coef2 * x_t

            if t_step > 0:
                noise = torch.randn_like(x_t)
                x_t = mean + torch.sqrt(posterior_variance) * noise
            else:
                x_t = mean

        future_world = x_t * scale
        return future_world

    # -----------------------------
    # 5) Debug report
    # -----------------------------
    @torch.no_grad()
    def debug_report_world(loader, max_B=8):
        batch = next(iter(loader))

        past_norm = batch["past"][:max_B]
        future_norm = batch["future"][:max_B]

        past_world = past_norm.clone() * scale
        real_world = future_norm.clone() * scale
        samp_world = sample_future_denorm(batch, max_B=max_B)

        past_np = past_world.cpu().numpy()
        real_np = real_world.cpu().numpy()
        samp_np = samp_world.cpu().numpy()

        print("=== SHAPES ===")
        print("past:", past_np.shape)
        print("real future:", real_np.shape)
        print("sampled future:", samp_np.shape)

        print("\\n=== NaN / Inf CHECK ===")
        print("sample has NaN:", np.isnan(samp_np).any())
        print("sample has Inf:", np.isinf(samp_np).any())

        print("\\n=== GLOBAL VALUE RANGE ===")
        print("past min/max:   ", float(np.min(past_np)), float(np.max(past_np)))
        print("real min/max:   ", float(np.min(real_np)), float(np.max(real_np)))
        print("sample min/max: ", float(np.min(samp_np)), float(np.max(samp_np)))

        print("\\n=== PER-CHANNEL MEAN ± STD ===")
        channel_names = ["x_i", "y_i", "x_j", "y_j"]
        for c, name in enumerate(channel_names):
            p_mean, p_std = np.mean(past_np[:, :, c]), np.std(past_np[:, :, c])
            r_mean, r_std = np.mean(real_np[:, :, c]), np.std(real_np[:, :, c])
            s_mean, s_std = np.mean(samp_np[:, :, c]), np.std(samp_np[:, :, c])

            print(f"{name}:")
            print(f"  past   mean={p_mean:.3f}, std={p_std:.3f}")
            print(f"  real   mean={r_mean:.3f}, std={r_std:.3f}")
            print(f"  sample mean={s_mean:.3f}, std={s_std:.3f}")

        print("\\n=== STEP-TO-STEP MOTION MAGNITUDE ===")
        real_step = np.linalg.norm(real_np[:, 1:, :] - real_np[:, :-1, :], axis=-1)
        samp_step = np.linalg.norm(samp_np[:, 1:, :] - samp_np[:, :-1, :], axis=-1)
        print(f"real   mean step norm: {np.mean(real_step):.6f}")
        print(f"sample mean step norm: {np.mean(samp_step):.6f}")

        print("\\n=== FUTURE START GAP FROM PAST END ===")
        past_end = past_np[:, -1, :]
        real_start_gap = np.linalg.norm(real_np[:, 0, :] - past_end, axis=-1)
        samp_start_gap = np.linalg.norm(samp_np[:, 0, :] - past_end, axis=-1)
        print(f"real   mean start gap: {np.mean(real_start_gap):.6f}")
        print(f"sample mean start gap: {np.mean(samp_start_gap):.6f}")

        print("\\n=== FINAL-POINT ERROR VS REAL FUTURE ===")
        final_err = np.linalg.norm(samp_np[:, -1, :] - real_np[:, -1, :], axis=-1)
        print(f"mean final-point error: {np.mean(final_err):.6f}")
        print(f"std  final-point error: {np.std(final_err):.6f}")

        print("\\n=== SAMPLE EXAMPLE (example 0) ===")
        print("past last 3 steps:")
        print(np.round(past_np[0, -3:, :], 6))
        print("real future first 5 steps:")
        print(np.round(real_np[0, :5, :], 6))
        print("sample future first 5 steps:")
        print(np.round(samp_np[0, :5, :], 6))

    # -----------------------------
    # 6) Quick sampling test
    # -----------------------------
    def quick_sample_test():
        test_batch = next(iter(loader))
        future_sample_world = sample_future_denorm(test_batch, max_B=4)
        print("sampled future (world) shape:", future_sample_world.shape)
        print("first sample, first 3 steps:")
        print(future_sample_world[0, :3].cpu().numpy())

