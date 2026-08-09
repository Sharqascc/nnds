
import numpy as np
import torch

def load_eval_model(checkpoint_path, device, T=15, N=1, F=4, cond_dim=4,
                    hidden_dim=128):
    from traffic_diffusion.trajectory_diffusion import TrajectoryDiffusionModel

    traj_shape = (T, N, F)
    model = TrajectoryDiffusionModel(
        traj_shape=traj_shape,
        cond_dim=cond_dim,
        hidden_dim=hidden_dim,
    ).to(device)

    # weights_only=False: checkpoints may bundle a "stats" dict (numpy arrays)
    # alongside the state_dict. Only load checkpoints you trust.
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()
    return model

def sample_future(model, loader, device, T=15, N=1, F=4,
                  num_samples=10, num_steps=50):
    all_samples = []
    with torch.no_grad():
        for x0_batch, cond_batch in loader:
            B = x0_batch.shape[0]
            cond_batch = cond_batch.to(device)

            cond_rep = cond_batch.repeat(num_samples, 1)  # (S*B, cond_dim)
            x_samples = model.sample(cond_rep, num_steps=num_steps)  # (S*B, T, N, F)
            x_samples = x_samples.view(num_samples, B, T, N, F)
            x_samples_flat = x_samples.cpu().numpy().reshape(num_samples, B, -1)
            all_samples.append(x_samples_flat)

    if not all_samples:
        return None
    return np.concatenate(all_samples, axis=1)  # (S, total_B, T*N*F)
