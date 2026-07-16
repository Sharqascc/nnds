
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class TrajectoryDataset(Dataset):
    def __init__(self, trajectories: torch.Tensor, conditions: torch.Tensor):
        assert trajectories.shape[0] == conditions.shape[0]
        self.trajectories = trajectories
        self.conditions = conditions

    def __len__(self):
        return self.trajectories.shape[0]

    def __getitem__(self, idx):
        return self.trajectories[idx], self.conditions[idx]

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


class SimpleUNet1D(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.fc_cond = nn.Linear(cond_dim, dim)
        self.time_embed = SinusoidalTimeEmbedding(dim)
        self.fc_time = nn.Linear(dim, dim)
        self.net = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.SiLU(),
            nn.Linear(2 * dim, 2 * dim),
            nn.SiLU(),
            nn.Linear(2 * dim, dim),
        )

    def forward(self, x, cond, t):
        c = self.fc_cond(cond)
        te = self.fc_time(self.time_embed(t))
        h = x + c + te
        return self.net(h)

class TrajectoryDiffusionModel(nn.Module):
    def __init__(self, traj_shape, cond_dim, num_steps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        super().__init__()
        self.T, self.N, self.F = traj_shape
        self.D = self.T * self.N * self.F
        self.cond_dim = cond_dim
        self.num_steps = num_steps
        self.device = device

        betas = torch.linspace(beta_start, beta_end, num_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

        self.model = SimpleUNet1D(dim=self.D, cond_dim=cond_dim).to(self.device)

    def q_sample(self, x0, t, noise=None):
        x0 = x0.reshape(x0.shape[0], -1)
        if noise is None:
            noise = torch.randn_like(x0)
        else:
            noise = noise.reshape(noise.shape[0], -1)
        a_t = self.alphas_cumprod[t].view(-1, 1)
        return torch.sqrt(a_t) * x0 + torch.sqrt(1.0 - a_t) * noise

    def p_losses(self, x0, cond, t):
        x0 = x0.reshape(x0.shape[0], -1)
        cond = cond.reshape(cond.shape[0], -1)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        noise_pred = self.model(xt, cond, t)
        return nn.functional.mse_loss(noise_pred, noise)

    def forward(self, x0, cond):
        b = x0.shape[0]
        t = torch.randint(0, self.num_steps, (b,), device=self.device, dtype=torch.long)
        return self.p_losses(x0, cond, t)

    @torch.no_grad()
    def sample(self, cond, num_steps=None):
        steps = self.num_steps if num_steps is None else min(num_steps, self.num_steps)
        b = cond.shape[0]
        x = torch.randn(b, self.D, device=self.device)

        for i in reversed(range(steps)):
            t = torch.full((b,), i, device=self.device, dtype=torch.long)
            noise_pred = self.model(x, cond, t)

            alpha = self.alphas[i]
            alpha_bar = self.alphas_cumprod[i]
            beta = self.betas[i]

            if i > 0:
                z = torch.randn_like(x)
            else:
                z = torch.zeros_like(x)

            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * noise_pred) + torch.sqrt(beta) * z

        return x.view(b, self.T, self.N, self.F)
