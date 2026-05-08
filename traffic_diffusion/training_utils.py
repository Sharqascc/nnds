from __future__ import annotations
"""
Training utilities for trajectory diffusion models.

Provides:
- Data cleaning and normalization
- Train/eval split with stratification
- Model factory with validation
- Training loop with gradient clipping and early stopping
- Optional LR scheduler, data augmentation, gradient accumulation
- Optional wandb experiment tracking and mixed precision training
- Integration with nnds.core.types (optional)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence, Tuple, Dict, Any, Optional, Union, List
from dataclasses import dataclass

import torch
import numpy as np
from torch.utils.data import DataLoader, random_split, Subset

# Import core types (once you have them)
try:
    from nnds.core.types import TrajectoryBatch, DiffusionDatasetLike
    CORE_TYPES_AVAILABLE = True
except ImportError:
    CORE_TYPES_AVAILABLE = False
    warnings.warn("nnds.core.types not found. Using fallback types.")

from traffic_diffusion.trajectory_diffusion import (
    TrajectoryDiffusionModel,
    TrajectoryDataset,
)

# Try wandb (optional)
try:
    import wandb  # type: ignore
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
@dataclass
class TrainingConfig:
    """Configuration for diffusion model training."""
    batch_size: int = 14
    learning_rate: float = 1e-3
    num_epochs: int = 50

    # Optimization / stability
    gradient_clip: float = 1.0
    gradient_accumulation_steps: int = 1

    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4

    # Data split
    train_ratio: float = 0.8
    seed: int = 42

    # Checkpointing
    save_best: bool = True
    save_last: bool = True
    checkpoint_dir: str = "checkpoints"

    # LR scheduler (ReduceLROnPlateau)
    use_scheduler: bool = False
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5

    # Data augmentation
    aug_noise_std: float = 0.0
    aug_dropout_prob: float = 0.0

    # Mixed precision
    use_amp: bool = False

    # Experiment tracking
    use_wandb: bool = False
    wandb_project: str = "nnds_diffusion"
    wandb_run_name: Optional[str] = None


# ---------------------------------------------------------------------
# Validation / cleaning / normalization
# ---------------------------------------------------------------------
def validate_trajectory_data(
    x0: torch.Tensor,
    cond: torch.Tensor,
    allow_nan: bool = False,
) -> Tuple[bool, str]:
    """
    Validate trajectory data shapes and values.

    Args:
        x0: Trajectory tensor
        cond: Condition tensor
        allow_nan: Whether to allow NaN/Inf values

    Returns:
        (is_valid, error_message)
    """
    if x0.dim() < 2:
        return False, f"x0 has {x0.dim()} dimensions, expected at least 2"

    if cond.dim() < 2:
        return False, f"cond has {cond.dim()} dimensions, expected at least 2"

    if x0.shape[0] != cond.shape[0]:
        return False, f"Batch size mismatch: x0={x0.shape[0]}, cond={cond.shape[0]}"

    if not allow_nan:
        if torch.isnan(x0).any():
            return False, "NaN values found in x0"
        if torch.isnan(cond).any():
            return False, "NaN values found in cond"
        if torch.isinf(x0).any():
            return False, "Inf values found in x0"
        if torch.isinf(cond).any():
            return False, "Inf values found in cond"

    return True, ""


def clean_nonfinite(tensor: torch.Tensor, fill_value: float = 0.0) -> torch.Tensor:
    """Replace NaN and Inf with fill_value."""
    tensor = tensor.clone()
    tensor[~torch.isfinite(tensor)] = fill_value
    return tensor


def normalize_tensor(
    tensor: torch.Tensor,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize tensor to zero mean and unit variance.

    Args:
        tensor: Input tensor
        mean: Precomputed mean (optional)
        std: Precomputed std (optional)
        eps: Small constant for numerical stability

    Returns:
        (normalized_tensor, mean, std)
    """
    if mean is None:
        mean = tensor.mean(dim=0, keepdim=True)
    if std is None:
        std = tensor.std(dim=0, keepdim=True).clamp_min(eps)

    normalized = (tensor - mean) / std
    return normalized, mean, std


# ---------------------------------------------------------------------
# Dataset splitting
# ---------------------------------------------------------------------
def stratified_split(
    dataset: torch.utils.data.Dataset,
    train_ratio: float = 0.8,
    seed: int = 42,
    stratify_by: Optional[torch.Tensor] = None,
) -> Tuple[Subset, Subset]:
    """
    Split dataset with optional stratification.

    Args:
        dataset: Full dataset
        train_ratio: Proportion for training
        seed: Random seed
        stratify_by: Optional tensor for stratified split

    Returns:
        (train_dataset, eval_dataset)
    """
    n_total = len(dataset)
    n_train = int(train_ratio * n_total)
    n_eval = n_total - n_train

    # Ensure both splits have at least one sample
    if n_train == 0:
        n_train = 1
        n_eval = n_total - 1
    if n_eval == 0:
        n_eval = 1
        n_train = n_total - 1

    if stratify_by is not None:
        unique_classes = torch.unique(stratify_by)
        train_indices: List[int] = []
        eval_indices: List[int] = []

        for cls in unique_classes:
            cls_indices = torch.where(stratify_by == cls)[0].tolist()
            n_cls_train = int(len(cls_indices) * train_ratio)
            train_indices.extend(cls_indices[:n_cls_train])
            eval_indices.extend(cls_indices[n_cls_train:])

        generator = torch.Generator().manual_seed(seed)
        train_indices = torch.tensor(train_indices)[
            torch.randperm(len(train_indices), generator=generator)
        ].tolist()
        eval_indices = torch.tensor(eval_indices)[
            torch.randperm(len(eval_indices), generator=generator)
        ].tolist()

        train_dataset = Subset(dataset, train_indices)
        eval_dataset = Subset(dataset, eval_indices)
    else:
        generator = torch.Generator().manual_seed(seed)
        train_dataset, eval_dataset = random_split(
            dataset, [n_train, n_eval], generator=generator
        )

    return train_dataset, eval_dataset


# ---------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------
def augment_trajectory(
    x0: torch.Tensor,
    noise_std: float = 0.01,
    dropout_prob: float = 0.1,
) -> torch.Tensor:
    """Add small noise or dropout for data augmentation."""
    if noise_std > 0:
        x0 = x0 + torch.randn_like(x0) * noise_std
    if dropout_prob > 0:
        mask = torch.bernoulli(torch.ones_like(x0) * (1 - dropout_prob))
        x0 = x0 * mask
    return x0


# ---------------------------------------------------------------------
# Dataloaders
# ---------------------------------------------------------------------
def build_clean_dataloaders(
    raw_dataset: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    batch_size: int = 14,
    T: int = 15,
    N: int = 1,
    F: int = 4,
    train_ratio: float = 0.8,
    seed: int = 42,
    stratify: bool = False,
) -> Tuple[
    DataLoader,
    DataLoader,
    TrajectoryDataset,
    TrajectoryDataset,
    Dict[str, Any],
]:
    """
    Build cleaned and normalized dataloaders for diffusion training.

    Args:
        raw_dataset: Sequence of (x0, cond) tensors
        batch_size: Batch size for dataloaders
        T: Time steps (for validation / info)
        N: Number of actors (for validation / info)
        F: Feature dimension (for validation / info)
        train_ratio: Proportion of data for training
        seed: Random seed for reproducibility
        stratify: Whether to use stratified split (requires labels)

    Returns:
        train_loader, eval_loader, train_dataset, eval_dataset, normalization_stats
    """
    if len(raw_dataset) == 0:
        raise ValueError("raw_dataset is empty")

    x0_all = torch.stack([d[0] for d in raw_dataset])
    cond_all = torch.stack([d[1] for d in raw_dataset])

    valid, msg = validate_trajectory_data(x0_all, cond_all, allow_nan=True)
    if not valid:
        raise ValueError(f"Invalid input data: {msg}")

    x0_clean = clean_nonfinite(x0_all)
    cond_clean = clean_nonfinite(cond_all)

    valid, msg = validate_trajectory_data(x0_clean, cond_clean, allow_nan=False)
    if not valid:
        raise RuntimeError(f"Data still invalid after cleaning: {msg}")

    x0_norm, x_mean, x_std = normalize_tensor(x0_clean)
    cond_norm, c_mean, c_std = normalize_tensor(cond_clean)

    dataset = TrajectoryDataset(trajectories=x0_norm, conditions=cond_norm)

    if stratify and hasattr(raw_dataset, "labels"):
        labels = torch.tensor([getattr(d, "label", 0) for d in raw_dataset])
        train_dataset, eval_dataset = stratified_split(
            dataset, train_ratio=train_ratio, seed=seed, stratify_by=labels
        )
    else:
        train_dataset, eval_dataset = stratified_split(
            dataset, train_ratio=train_ratio, seed=seed, stratify_by=None
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    stats: Dict[str, Any] = {
        "x_mean": x_mean,
        "x_std": x_std,
        "c_mean": c_mean,
        "c_std": c_std,
        "T": T,
        "N": N,
        "F": F,
        "train_size": len(train_dataset),
        "eval_size": len(eval_dataset),
    }

    print(f"✅ Built dataloaders: train={len(train_dataset)}, eval={len(eval_dataset)}")

    return train_loader, eval_loader, train_dataset, eval_dataset, stats


# ---------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------
def create_model(
    device: torch.device,
    T: int = 15,
    N: int = 1,
    F: int = 4,
    cond_dim: int = 4,
    num_steps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> TrajectoryDiffusionModel:
    """
    Factory for TrajectoryDiffusionModel with validation.
    """
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    if N <= 0:
        raise ValueError(f"N must be positive, got {N}")
    if F <= 0:
        raise ValueError(f"F must be positive, got {F}")
    if cond_dim <= 0:
        raise ValueError(f"cond_dim must be positive, got {cond_dim}")
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")

    traj_shape = (T, N, F)

    model_kwargs = model_kwargs or {}
    model = TrajectoryDiffusionModel(
        traj_shape=traj_shape,
        cond_dim=cond_dim,
        num_steps=num_steps,
        beta_start=beta_start,
        beta_end=beta_end,
        device=device,
        **model_kwargs,
    ).to(device)

    print(f"✅ Created model with traj_shape={traj_shape}, cond_dim={cond_dim}")
    return model


# ---------------------------------------------------------------------
# Optional wandb logger
# ---------------------------------------------------------------------
def setup_logger(
    project_name: str = "nnds_diffusion",
    config: Optional[TrainingConfig] = None,
    run_name: Optional[str] = None,
):
    """Optional integration with wandb."""
    if not WANDB_AVAILABLE:
        return None
    wandb.init(project=project_name, config=None if config is None else vars(config),
               name=run_name)
    return wandb


# ---------------------------------------------------------------------
# Training loop (with scheduler, aug, grad accumulation, AMP)
# ---------------------------------------------------------------------
def train_diffusion_model(
    model: TrajectoryDiffusionModel,
    train_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 50,
    save_dir: str = "checkpoints",
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr: float = 1e-3,
    gradient_clip: float = 1.0,
    eval_loader: Optional[DataLoader] = None,
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 1e-4,
    verbose: bool = True,
    # new optional knobs (if you do not use TrainingConfig directly)
    use_scheduler: bool = False,
    scheduler_patience: int = 5,
    scheduler_factor: float = 0.5,
    gradient_accumulation_steps: int = 1,
    aug_noise_std: float = 0.0,
    aug_dropout_prob: float = 0.0,
    use_amp: bool = False,
    use_wandb: bool = False,
    wandb_project: str = "nnds_diffusion",
    wandb_run_name: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], List[Dict[str, Any]]]:
    """
    Enhanced training loop with gradient clipping, validation, early stopping,
    LR scheduler, augmentation, gradient accumulation, AMP, and optional wandb.
    """
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=scheduler_patience,
            factor=scheduler_factor,
            verbose=verbose,
        )

    scaler: Optional[torch.cuda.amp.GradScaler] = None
    if use_amp and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()

    wandb_logger = None
    if use_wandb:
        wandb_logger = setup_logger(wandb_project, None, wandb_run_name)

    best_loss = float("inf")
    best_ckpt_path: Optional[str] = None
    last_ckpt_path: Optional[str] = None
    history: List[Dict[str, Any]] = []

    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        num_batches = 0

        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_loader):
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x0_batch, cond_batch = batch
            else:
                x0_batch = batch[0]
                cond_batch = batch[1]

            x0_batch = x0_batch.to(device)
            cond_batch = cond_batch.to(device)

            # Data augmentation (on x0 only)
            if aug_noise_std > 0 or aug_dropout_prob > 0:
                x0_batch = augment_trajectory(
                    x0_batch,
                    noise_std=aug_noise_std,
                    dropout_prob=aug_dropout_prob,
                )

            # Forward + loss (with or without AMP)
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    loss = model(x0_batch, cond_batch)
            else:
                loss = model(x0_batch, cond_batch)

            if torch.isnan(loss) or torch.isinf(loss):
                if verbose:
                    print(f"⚠️ NaN/Inf loss at epoch {epoch}, batch {batch_idx}, skipping")
                continue

            # Gradient accumulation
            effective_loss = loss / max(gradient_accumulation_steps, 1)

            if scaler is not None:
                scaler.scale(effective_loss).backward()
            else:
                effective_loss.backward()

            # Step when enough micro-batches accumulated
            if (batch_idx + 1) % max(gradient_accumulation_steps, 1) == 0:
                if gradient_clip > 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

            train_loss += float(loss.item())
            num_batches += 1

        train_loss /= max(num_batches, 1)

        # Validation
        eval_loss = None
        if eval_loader is not None:
            model.eval()
            eval_loss = 0.0
            num_eval_batches = 0

            with torch.no_grad():
                for batch in eval_loader:
                    if isinstance(batch, (list, tuple)) and len(batch) == 2:
                        x0_batch, cond_batch = batch
                    else:
                        x0_batch = batch[0]
                        cond_batch = batch[1]

                    x0_batch = x0_batch.to(device)
                    cond_batch = cond_batch.to(device)

                    if scaler is not None:
                        with torch.cuda.amp.autocast():
                            loss = model(x0_batch, cond_batch)
                    else:
                        loss = model(x0_batch, cond_batch)

                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        eval_loss += float(loss.item())
                        num_eval_batches += 1

            eval_loss /= max(num_eval_batches, 1)

        # LR scheduler step
        if scheduler is not None and eval_loader is not None and eval_loss is not None:
            scheduler.step(eval_loss)

        epoch_record: Dict[str, Any] = {"epoch": epoch, "train_loss": train_loss}
        if eval_loss is not None:
            epoch_record["eval_loss"] = eval_loss
        history.append(epoch_record)

        if verbose:
            log_msg = f"Epoch {epoch:03d}/{num_epochs} | train_loss={train_loss:.6f}"
            if eval_loss is not None:
                log_msg += f" | eval_loss={eval_loss:.6f}"
            if scheduler is not None:
                current_lr = optimizer.param_groups[0]["lr"]
                log_msg += f" | lr={current_lr:.2e}"
            print(log_msg)

        # wandb logging
        if wandb_logger is not None:
            log_dict = {
                "epoch": epoch,
                "train_loss": train_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            if eval_loss is not None:
                log_dict["eval_loss"] = eval_loss
            wandb_logger.log(log_dict)

        # Save last checkpoint
        last_path = save_dir_path / "traj_diffusion_last.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": train_loss,
            },
            last_path,
        )
        last_ckpt_path = str(last_path)

        # Early stopping logic
        current_loss = eval_loss if eval_loss is not None else train_loss

        if current_loss < best_loss - early_stopping_min_delta:
            best_loss = current_loss
            patience_counter = 0

            best_path = save_dir_path / "traj_diffusion_best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                },
                best_path,
            )
            best_ckpt_path = str(best_path)
            if verbose:
                print(f"  ✅ Saved best checkpoint (loss={best_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience and eval_loader is not None:
                if verbose:
                    print(
                        f"⏹️ Early stopping at epoch {epoch} "
                        f"(no improvement for {early_stopping_patience} epochs)"
                    )
                break

    if verbose:
        print("\n✅ Training complete")
        if best_ckpt_path:
            print(f"   Best checkpoint: {best_ckpt_path}")
        if last_ckpt_path:
            print(f"   Last checkpoint: {last_ckpt_path}")

    if wandb_logger is not None:
        wandb_logger.finish()

    return best_ckpt_path, last_ckpt_path, history


# ---------------------------------------------------------------------
# Backward compatibility wrapper
# ---------------------------------------------------------------------
def train_diffusion_model_simple(
    model: TrajectoryDiffusionModel,
    train_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 50,
    save_dir: str = "checkpoints",
) -> Tuple[Optional[str], Optional[str], List[Dict[str, Any]]]:
    """
    Simple training wrapper for backward compatibility with original interface.
    """
    return train_diffusion_model(
        model=model,
        train_loader=train_loader,
        device=device,
        num_epochs=num_epochs,
        save_dir=save_dir,
        gradient_clip=1.0,
        eval_loader=None,
    )


# ---------------------------------------------------------------------
# Optional typed helpers
# ---------------------------------------------------------------------
if CORE_TYPES_AVAILABLE:
    from nnds.core.types import TrajectoryBatch

    def batch_to_trajectory_batch(
        x0: torch.Tensor,
        cond: torch.Tensor,
        fps: float = 30.0,
        meta: Optional[Dict[str, Any]] = None,
    ) -> TrajectoryBatch:
        """Convert tensors to TrajectoryBatch for typed workflows."""
        return TrajectoryBatch(
            inputs=x0,
            targets=cond,
            meta=meta or {},
            fps=fps,
        )
