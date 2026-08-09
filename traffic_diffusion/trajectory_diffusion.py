import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """Simple (trajectory, condition) pair dataset for diffusion training.

    Args:
        trajectories: (num_samples, Th, N_agents, dim) tensor of trajectories.
        conditions:   (num_samples, cond_dim) tensor of conditioning vectors.
    """

    def __init__(self, trajectories: torch.Tensor, conditions: torch.Tensor):
        if trajectories.shape[0] != conditions.shape[0]:
            raise ValueError(
                "trajectories and conditions must have the same number of "
                f"samples, got {trajectories.shape[0]} vs {conditions.shape[0]}"
            )
        self.trajectories = trajectories
        self.conditions = conditions

    def __len__(self):
        return self.trajectories.shape[0]

    def __getitem__(self, idx):
        return self.trajectories[idx], self.conditions[idx]


class TemporalResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)

    def forward(self, x, emb):
        res = x
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + emb.unsqueeze(-1)
        h = self.norm2(self.conv2(h))
        return F.silu(h + res)

class TrajectoryDiffusionModel(nn.Module):
    def __init__(self, traj_shape=(16, 1, 2), cond_dim=4, hidden_dim=128):
        super().__init__()
        self.Th, self.N_agents, self.dim = traj_shape
        self.input_dim = self.dim * self.N_agents
        self.hidden_dim = hidden_dim
        
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.input_proj = nn.Conv1d(self.input_dim, hidden_dim, kernel_size=3, padding=1)
        
        self.blocks = nn.ModuleList([
            TemporalResBlock(hidden_dim),
            TemporalResBlock(hidden_dim),
            TemporalResBlock(hidden_dim),
            TemporalResBlock(hidden_dim)
        ])
        
        self.out_proj = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, self.input_dim, kernel_size=3, padding=1)
        )

    def forward(self, x, cond, t):
        B = x.shape[0]
        x_flat = x.reshape(B, self.Th, self.input_dim).permute(0, 2, 1)
        
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        if t.max() > 1.0:
            t = t / 100.0
            
        t_emb = self.time_mlp(t)
        
        if cond.ndim > 2:
            cond = cond.reshape(B, -1)
        c_emb = self.cond_mlp(cond)
        
        emb = t_emb + c_emb
        
        h = self.input_proj(x_flat)
        for block in self.blocks:
            h = block(h, emb)
            
        out = self.out_proj(h)
        out = out.permute(0, 2, 1).reshape(B, self.Th, self.N_agents, self.dim)
        return out

    def compute_loss(self, x0, cond):
        """Rectified Flow Matching velocity field loss: v = noise - x0"""
        B = x0.shape[0]
        device = x0.device
        
        t = torch.rand(B, 1, device=device) # Timestamps in [0, 1]
        noise = torch.randn_like(x0)
        
        t_exp = t.view(B, 1, 1, 1)
        xt = (1.0 - t_exp) * x0 + t_exp * noise
        
        v_target = noise - x0
        pred_v = self(xt, cond, t)
        
        return F.mse_loss(pred_v, v_target)

    @torch.no_grad()
    def sample(self, cond, num_steps=50):
        """Stable Euler ODE integration from noise (t=1.0) to velocity data (t=0.0)."""
        B = cond.shape[0]
        device = cond.device
        
        x = torch.randn(B, self.Th, self.N_agents, self.dim, device=device)
        dt = 1.0 / num_steps
        
        for step in reversed(range(1, num_steps + 1)):
            t_val = step / num_steps
            t = torch.full((B, 1), t_val, device=device)
            
            v_pred = self(x, cond, t)
            x = x - v_pred * dt
            
        return x
