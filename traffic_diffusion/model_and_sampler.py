import torch
import numpy as np

from traffic_diffusion.trajectory_diffusion import TrajectoryDiffusionModel

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL = None

def load_model(checkpoint_path: str, traj_shape, cond_dim, num_steps=1000):
    """
    Construct TrajectoryDiffusionModel and load weights from checkpoint.

    traj_shape: (T, N, F) used during training.
    cond_dim:   conditioning dimension used during training.
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    model = TrajectoryDiffusionModel(
        traj_shape=traj_shape,
        cond_dim=cond_dim,
        num_steps=num_steps,
        device=_DEVICE,
    )

    ckpt = torch.load(checkpoint_path, map_location=_DEVICE)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state)
    model.to(_DEVICE).eval()
    _MODEL = model
    return _MODEL


@torch.no_grad()
def sample_future_denorm(batch, checkpoint_path: str, num_samples: int = 1,
                         traj_shape=(9, 2, 2), cond_dim=4, num_steps=1000):
    """
    Inputs:
      batch:
        - 'past':   (B, Th, 4) normalized input trajectories (2 agents x (x,y))
        - 'future': (B, Tf, 4) normalized future (only used to know Tf)
      checkpoint_path: path to trained diffusion model checkpoint.
    Returns:
      samples_world: (num_samples, B, Tf, 4) in normalized coordinates
                     (if you add de-normalization, convert here).
    """
    # Derive shapes
    B, Th, D = batch["past"].shape
    Tf = batch["future"].shape[1]  # future horizon
    # traj_shape = (T, N, F) where T = Tf, N=2 agents, F=2 coords
    T = Tf
    N = 2
    F = 2
    traj_shape = (T, N, F)

    # Load model once
    model = load_model(checkpoint_path, traj_shape=traj_shape, cond_dim=cond_dim, num_steps=num_steps)

    # Build conditioning vector(s) from batch["past"]
    # Here, a simple example: flattened last past step for each agent.
    past = batch["past"].to(_DEVICE)  # (B, Th, 4)
    # Example cond: final past positions (x_i, y_i, x_j, y_j)
    cond = past[:, -1, :]  # (B, 4)
    cond = cond.to(_DEVICE)

    samples = []
    for _ in range(num_samples):
        # model.sample expects cond shape (B, cond_dim) and returns (B, T, N, F)
        x = model.sample(cond)  # (B, T, N, F)
        x = x.to("cpu").numpy()  # -> (B, T, N, F)
        # reshape to (B, T, 4) with agents concatenated along feature dim
        x_4 = x.reshape(B, T, N * F)  # (B, Tf, 4)
        samples.append(x_4)

    samples_np = np.stack(samples, axis=0)  # (S, B, Tf, 4)
    return samples_np
