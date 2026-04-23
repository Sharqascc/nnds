
import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from traffic_diffusion.trajectory_diffusion import TrajectoryDataset, TrajectoryDiffusionModel


def parse_traj_txy(cell):
    arr = np.array(ast.literal_eval(cell), dtype=float)  # (T,3): t,x,y
    return arr[:, 0], arr[:, 1:3]


def build_training_tensors(csv_path, Th=8):
    df = pd.read_csv(csv_path)

    x0_list = []
    cond_list = []

    for ci, cj in zip(df["world_traj_i"], df["world_traj_j"]):
        t_i, xy_i = parse_traj_txy(ci)
        t_j, xy_j = parse_traj_txy(cj)

        T = min(len(xy_i), len(xy_j))
        if T < (Th + 2):
            continue

        xy_i = xy_i[:T]
        xy_j = xy_j[:T]

        past_i = xy_i[:Th]
        past_j = xy_j[:Th]
        fut_i = xy_i[Th:]
        fut_j = xy_j[Th:]

        Tf = min(len(fut_i), len(fut_j))
        if Tf < 2:
            continue

        fut_i = fut_i[:Tf]
        fut_j = fut_j[:Tf]

        cx, cy = past_i[-1]
        cond = np.array([
            past_i[-1, 0] - cx, past_i[-1, 1] - cy,
            past_j[-1, 0] - cx, past_j[-1, 1] - cy,
        ], dtype=np.float32)

        fut_i_norm = fut_i - np.array([cx, cy], dtype=np.float32)
        fut_j_norm = fut_j - np.array([cx, cy], dtype=np.float32)

        x0 = np.stack([fut_i_norm, fut_j_norm], axis=1)  # (Tf, 2, 2)

        x0_list.append(x0.astype(np.float32))
        cond_list.append(cond)

    if not x0_list:
        raise RuntimeError("No valid training samples built from CSV")

    Tf_target = 9  # must match evaluator's Tf

    x0_list_fixed = []
    for x in x0_list:
        if x.shape[0] >= Tf_target:
            x_fixed = x[:Tf_target]
        else:
            pad = np.repeat(x[-1:, :, :], Tf_target - x.shape[0], axis=0)
            x_fixed = np.concatenate([x, pad], axis=0)
        x0_list_fixed.append(x_fixed.astype(np.float32))

    x0 = torch.tensor(np.stack(x0_list_fixed, axis=0), dtype=torch.float32)   # (B,9,2,2)
    cond = torch.tensor(np.stack(cond_list, axis=0), dtype=torch.float32)    # (B,4)

    return x0, cond


def train(
    csv_path="outputs/petevents_20_frames_test.csv",
    checkpoint_dir="checkpoints",
    Th=8,
    batch_size=32,
    epochs=1,
    lr=1e-3,
    num_steps=1000,
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    x0, cond = build_training_tensors(csv_path, Th=Th)
    print("x0 shape:", tuple(x0.shape))
    print("cond shape:", tuple(cond.shape))

    dataset = TrajectoryDataset(x0, cond)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    traj_shape = tuple(x0.shape[1:])   # (Tf,2,2)
    cond_dim = cond.shape[1]

    model = TrajectoryDiffusionModel(
        traj_shape=traj_shape,
        cond_dim=cond_dim,
        num_steps=num_steps,
        device=device,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    best_path = os.path.join(checkpoint_dir, "traj_diffusion_best.pt")
    last_path = os.path.join(checkpoint_dir, "traj_diffusion_last.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        count = 0

        for batch_x0, batch_cond in loader:
            batch_x0 = batch_x0.to(device)
            batch_cond = batch_cond.to(device)

            batch_x0_flat = batch_x0.view(batch_x0.shape[0], -1)

            optimizer.zero_grad()
            loss = model(batch_x0_flat, batch_cond)
            loss.backward()
            optimizer.step()

            running += float(loss.item()) * batch_x0.shape[0]
            count += batch_x0.shape[0]

        epoch_loss = running / max(count, 1)
        print(f"Epoch {epoch:03d}/{epochs} | loss={epoch_loss:.6f}")

        torch.save({
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "traj_shape": traj_shape,
            "cond_dim": int(cond_dim),
            "num_steps": int(num_steps),
            "th": int(Th),
        }, last_path)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "traj_shape": traj_shape,
                "cond_dim": int(cond_dim),
                "num_steps": int(num_steps),
                "th": int(Th),
                "best_loss": float(best_loss),
            }, best_path)
            print("Saved best checkpoint to:", best_path)

    print("Training complete.")
    print("Best checkpoint:", best_path)
    print("Last checkpoint:", last_path)


if __name__ == "__main__":
    train()
