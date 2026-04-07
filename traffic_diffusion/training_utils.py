
import torch
from torch.utils.data import DataLoader, random_split
from traffic_diffusion.trajectory_diffusion import TrajectoryDiffusionModel, TrajectoryDataset

def build_clean_dataloaders(raw_dataset, batch_size=14, T=15, N=1, F=4):
    x0_all = torch.stack([d[0] for d in raw_dataset])
    cond_all = torch.stack([d[1] for d in raw_dataset])

    x0_clean = x0_all.clone()
    cond_clean = cond_all.clone()
    x0_clean[~torch.isfinite(x0_clean)] = 0.0
    cond_clean[~torch.isfinite(cond_clean)] = 0.0

    x_mean = x0_clean.mean(dim=0, keepdim=True)
    x_std = x0_clean.std(dim=0, keepdim=True).clamp_min(1e-6)
    x0_clean = (x0_clean - x_mean) / x_std

    c_mean = cond_clean.mean(dim=0, keepdim=True)
    c_std = cond_clean.std(dim=0, keepdim=True).clamp_min(1e-6)
    cond_clean = (cond_clean - c_mean) / c_std

    dataset = TrajectoryDataset(trajectories=x0_clean, conditions=cond_clean)

    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_eval = n_total - n_train
    if n_eval == 0:
        n_train = n_total - 1
        n_eval = 1

    train_dataset, eval_dataset = random_split(
        dataset,
        [n_train, n_eval],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    stats = {
        "x_mean": x_mean,
        "x_std": x_std,
        "c_mean": c_mean,
        "c_std": c_std,
        "T": T,
        "N": N,
        "F": F,
    }
    return train_loader, eval_loader, train_dataset, eval_dataset, stats

def create_model(device, T=15, N=1, F=4, cond_dim=4,
                 num_steps=1000, beta_start=1e-4, beta_end=0.02):
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

def train_diffusion_model(model, train_loader, device,
                          num_epochs=50, save_dir="checkpoints",
                          optimizer=None, lr=1e-3):
    from pathlib import Path
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    best_ckpt_path = None
    last_ckpt_path = None
    history = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for x0_batch, cond_batch in train_loader:
            x0_batch = x0_batch.to(device)
            cond_batch = cond_batch.to(device)

            optimizer.zero_grad()
            loss = model(x0_batch, cond_batch)

            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"NaN/Inf loss at epoch {epoch}")

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        epoch_loss /= max(num_batches, 1)
        print(f"Epoch {epoch:03d}/{num_epochs} | loss={epoch_loss:.6f}")

        last_ckpt_path = save_dir / "traj_diffusion_last.pt"
        torch.save(model.state_dict(), last_ckpt_path)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_ckpt_path = save_dir / "traj_diffusion_best.pt"
            torch.save(model.state_dict(), best_ckpt_path)
            print("Saved best checkpoint to:", best_ckpt_path)

        history.append({"epoch": epoch, "loss": epoch_loss})

    print("Training complete.")
    print("Best checkpoint:", best_ckpt_path)
    print("Last checkpoint:", last_ckpt_path)
    return str(best_ckpt_path), str(last_ckpt_path), history
