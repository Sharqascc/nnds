
import numpy as np
import torch

def load_eval_model(checkpoint_path, device, T=15, N=1, F=4, cond_dim=4,
                    num_steps=1000, beta_start=1e-4, beta_end=0.02):
    from traffic_diffusion.trajectory_diffusion import TrajectoryDiffusionModel

    traj_shape = (T, N, F)
    model = TrajectoryDiffusionModel(
        traj_shape=traj_shape,
        cond_dim=cond_dim,
        num_steps=num_steps,
        beta_start=beta_start,
        beta_end=beta_end,
        device=device,
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def sample_future(model, loader, device, T=15, N=1, F=4,
                  num_samples=10, num_steps=None):
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
