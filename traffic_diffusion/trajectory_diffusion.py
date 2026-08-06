import torch
import torch.nn as nn
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))

class SimpleUNet(nn.Module):
    def __init__(self, traj_shape, cond_dim=2):
        super().__init__()
        self.Th, self.C, self.D = traj_shape
        traj_dim = self.Th * self.C * self.D
        input_dim = traj_dim + traj_dim + 1 # xt + condition + timestep
        
        hidden_dim = 512
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim)
        )
        
        self.output_proj = nn.Linear(hidden_dim, traj_dim)

    def forward(self, xt, cond, t):
        b, Th, c, d = xt.shape
        xt_flat = xt.reshape(b, -1)
        cond_flat = cond.reshape(b, -1)
        t_embed = t.float().unsqueeze(1) / 1000.0
        
        x_in = torch.cat([xt_flat, cond_flat, t_embed], dim=1)
        h = self.input_proj(x_in)
        h = self.res_blocks(h)
        out = self.output_proj(h)
        return out.reshape(b, Th, c, d)

class TrajectoryDiffusionModel(nn.Module):
    def __init__(self, traj_shape=(16, 1, 2), cond_dim=2, timesteps=100):
        super().__init__()
        self.traj_shape = traj_shape
        self.timesteps = timesteps
        self.model = SimpleUNet(traj_shape, cond_dim)

        beta = torch.linspace(1e-4, 0.02, timesteps)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)

    def forward(self, x0, cond):
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x0.device).long()
        noise = torch.randn_like(x0)
        
        alpha_bar_t = self.alpha_bar[t].view(b, 1, 1, 1)
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise
        
        noise_pred = self.model(xt, cond, t)
        return nn.functional.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample(self, cond):
        b = cond.shape[0]
        device = cond.device
        xt = torch.randn((b, *self.traj_shape), device=device)

        for i in reversed(range(self.timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            alpha_i = self.alpha[i]
            alpha_bar_i = self.alpha_bar[i]
            beta_i = self.beta[i]

            noise_pred = self.model(xt, cond, t)
            
            if i > 0:
                noise = torch.randn_like(xt)
            else:
                noise = torch.zeros_like(xt)

            coef = 1.0 / torch.sqrt(alpha_i)
            resid = beta_i / torch.sqrt(1.0 - alpha_bar_i)
            xt = coef * (xt - resid * noise_pred) + torch.sqrt(beta_i) * noise

        return xt
print("✅ Upgraded TrajectoryDiffusionModel with High-Capacity Residual Blocks.")
