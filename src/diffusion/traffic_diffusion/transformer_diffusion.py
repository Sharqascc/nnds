import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, T, C)
        return x + self.pe[: x.size(1), :].unsqueeze(0)


class TransformerTrajectoryDiffusion(nn.Module):
    """Transformer denoiser that predicts noise from noisy trajectory + condition."""

    def __init__(
        self,
        traj_dim=2,
        cond_dim=2,
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        max_len=32,
    ):
        super().__init__()
        self.traj_dim = traj_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        self.traj_proj = nn.Linear(traj_dim + cond_dim + 1, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(hidden_dim, traj_dim)

    def forward(self, noisy_traj, cond, t):
        B, T, _ = noisy_traj.shape
        t_norm = (t.float() / 1000.0).view(B, 1, 1).expand(-1, T, 1)
        x = torch.cat([noisy_traj, cond, t_norm], dim=-1)
        x = self.traj_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        noise_pred = self.out_proj(x)
        return noise_pred
