import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, x, emb):
        res = x
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + emb.unsqueeze(-1)
        h = self.norm2(self.conv2(h))
        return F.silu(h + res)

class TrajectoryDiffusionModel(nn.Module):
    def __init__(self, traj_shape=(16, 1, 2), cond_dim=2, hidden_dim=128):
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

    def p_losses(self, x0, cond):
        B = x0.shape[0]
        device = x0.device
        t = torch.rand(B, 1, device=device)
        noise = torch.randn_like(x0)
        
        t_exp = t.view(B, 1, 1, 1)
        xt = (1.0 - t_exp) * x0 + t_exp * noise
        
        pred_noise = self(xt, cond, t)
        return F.mse_loss(pred_noise, noise)

    def compute_loss(self, x0, cond):
        return self.p_losses(x0, cond)

    @torch.no_grad()
    def sample(self, cond, num_steps=20):
        B = cond.shape[0]
        device = cond.device
        x = torch.randn(B, self.Th, self.N_agents, self.dim, device=device)
        
        dt = 1.0 / num_steps
        for i in range(num_steps, 0, -1):
            t = torch.full((B, 1), i / num_steps, device=device)
            pred_noise = self(x, cond, t)
            x = x - pred_noise * dt
        return x
