import torch
import numpy as np

from traffic_diffusion.trajectory_diffusion import TrajectoryDiffusionModel

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL_CACHE = {}

def load_model(checkpoint_path: str, traj_shape, cond_dim, num_steps=1000):
    """
    Construct TrajectoryDiffusionModel and load weights from checkpoint.

    traj_shape: (T, N, F) used during training.
    cond_dim:   conditioning dimension used during training.

    Cached per (checkpoint_path, traj_shape, cond_dim, num_steps) so that
    loading a different checkpoint or config doesn't silently reuse a
    previously loaded model.
    """
    key = (checkpoint_path, tuple(traj_shape), cond_dim, num_steps)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

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
    _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


@torch.no_grad()
def sample_future_denorm(batch, checkpoint_path: str, num_samples: int = 1,
                         traj_shape=(9, 2, 2), cond_dim=4, num_steps=1000):
    """
    NOTE: distinct from traj_diffusion_normalized.sample_future_ddpm_loop, which
    runs an explicit DDPM reverse-diffusion loop step by step. This function
    instead loads a checkpointed model and delegates to its .sample() method,
    trusting that method to perform the reverse process correctly internally.

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
    T = Tf  # output future steps
    _, N, F = traj_shape  # use training shape metadata for agent/feature dims

    # traj_shape is passed as parameter: (9, 2, 2)
    # T is only the future horizon used for output reshaping

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
        x = model.sample(cond)  # (B, T_total, N, F) where T_total=9
        x = x[:, -T:, :, :]      # slice to future horizon only: (B, T, N, F) where T=5
        x = x.to("cpu").numpy()
        # reshape to (B, T, 4) with agents concatenated along feature dim
        x_4 = x.reshape(B, T, N * F)  # (B, Tf, 4)
        samples.append(x_4)

    samples_np = np.stack(samples, axis=0)  # (S, B, Tf, 4)
    return samples_np
