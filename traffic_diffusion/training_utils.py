from __future__ import annotations

from typing import Sequence, Tuple, Dict, Any

import torch
from torch.utils.data import DataLoader, random_split
from traffic_diffusion.trajectory_diffusion import TrajectoryDiffusionModel, TrajectoryDataset
# If you add nnds.core.types later, you can import TrajectoryBatch / DiffusionDatasetLike here.


def build_clean_dataloaders(
    raw_dataset: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    batch_size: int = 14,
    T: int = 15,
    N: int = 1,
    F: int = 4,
) -> Tuple[DataLoader, DataLoader, TrajectoryDataset, TrajectoryDataset, Dict[str, Any]]:
    """
    Take a raw (x0, cond) dataset, clean NaNs/Infs, normalize, and
    return train/eval loaders plus normalization stats.

    raw_dataset: sequence where each item is (x0, cond) tensors.
    """

    # Stack raw tensors into big arrays
    x0_all = torch.stack([d[0] for d in raw_dataset])  # (B, T, N, F) or similar
    cond_all = torch.stack([d[1] for d in raw_dataset])

    # Clone and clean non-finite values
    x0_clean = x0_all.clone()
    cond_clean = cond_all.clone()
    x0_clean[~torch.isfinite(x0_clean)] = 0.0
    cond_clean[~torch.isfinite(cond_clean)] = 0.0

    # Normalize trajectories
    x_mean = x0_clean.mean(dim=0, keepdim=True)
    x_std = x0_clean.std(dim=0, keepdim=True).clamp_min(1e-6)
    x0_clean = (x0_clean - x_mean) / x_std

    # Normalize conditions
    c_mean = cond_clean.mean(dim=0, keepdim=True)
    c_std = cond_clean.std(dim=0, keepdim=True).clamp_min(1e-6)
    cond_clean = (cond_clean - c_mean) / c_std

    # Wrap into your existing dataset class
    dataset = TrajectoryDataset(trajectories=x0_clean, conditions=cond_clean)

    # Train / eval split
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_eval = n_total - n_train
    if n_eval == 0:
        n_train = n_total - 1
        n_eval = 1

    train_dataset, eval_dataset = random_split(
        dataset,
        [n_train, n_eval],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    stats: Dict[str, Any] = {
        "x_mean": x_mean,
        "x_std": x_std,
        "c_mean": c_mean,
        "c_std": c_std,
        "T": T,
        "N": N,
        "F": F,
    }
    return train_loader, eval_loader, train_dataset, eval_dataset, stats


def create_model(
    device: torch.device,
    T: int = 15,
    N: int = 1,
    F: int = 4,
    cond_dim: int = 4,
    num_steps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> TrajectoryDiffusionModel:
    """
    Factory for TrajectoryDiffusionModel with explicit shape typing.
    """
    traj_shape = (T, N, F)
    model = TrajectoryDiffusionModel(
        traj_shape=traj_shape,
        cond_dim=cond_dim,
        num_steps=num_steps,
        beta_start=beta_start,
        beta_end=beta_end,
        device=device,
    ).to(device)
    return model


def train_diffusion_model(
    model: TrajectoryDiffusionModel,
    train_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 50,
    save_dir: str = "checkpoints",
    optimizer: torch.optim.Optimizer | None = None,
    lr: float = 1e-3,
) -> Tuple[str | None, str | None, list[Dict[str, Any]]]:
    """
    Standard training loop with checkpointing and basic numerical safety checks.
    """
    from pathlib import Path

    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    best_ckpt_path: str | None = None
    last_ckpt_path: str | None = None
    history: list[Dict[str, Any]] = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            # Support both plain (x0, cond) and Dataset returning dicts later if you want.
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x0_batch, cond_batch = batch
            else:
                # Future-proof: e.g. if you move to a TrajectoryBatch-like object
                x0_batch = batch[0]
                cond_batch = batch[1]

            x0_batch = x0_batch.to(device)
            cond_batch = cond_batch.to(device)

            optimizer.zero_grad()
            loss = model(x0_batch, cond_batch)

            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"NaN/Inf loss at epoch {epoch}")

            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            num_batches += 1

        epoch_loss /= max(num_batches, 1)
        print(f"Epoch {epoch:03d}/{num_epochs} | loss={epoch_loss:.6f}")

        # Last checkpoint
        last_path = save_dir_path / "traj_diffusion_last.pt"
        torch.save(model.state_dict(), last_path)
        last_ckpt_path = str(last_path)

        # Best checkpoint
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_path = save_dir_path / "traj_diffusion_best.pt"
            torch.save(model.state_dict(), best_path)
            best_ckpt_path = str(best_path)
            print("Saved best checkpoint to:", best_ckpt_path)

        history.append({"epoch": epoch, "loss": epoch_loss})

    print("Training complete.")
    print("Best checkpoint:", best_ckpt_path)
    print("Last checkpoint:", last_ckpt_path)
    return best_ckpt_path, last_ckpt_path, history
