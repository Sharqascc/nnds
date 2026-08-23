import os, sys
import numpy as np
import pandas as pd
import torch
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.stats import wasserstein_distance
from src.diffusion.traffic_diffusion.trajectory_diffusion import TrajectoryDiffusionModel

CKPT = "checkpoints_diffusion_del4_v4/traj_diffusion_best.pt"
CSV = "outputs/diffusion_del4_v4.csv"
TH = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_metrics(gen_trajs, gt_trajs, cond_trajs, real_pets, dt=0.1):
    errors = np.linalg.norm(gen_trajs - gt_trajs, axis=-1)
    ade = np.mean(errors)
    fde = np.mean(errors[:, -1])
    vel = np.diff(gen_trajs, axis=1) / dt
    acc = np.diff(vel, axis=1) / dt
    jerk = np.diff(acc, axis=1) / dt
    acc_viol = np.mean(np.linalg.norm(acc, axis=-1) > 4.5) * 100.0
    jerk_viol = np.mean(np.linalg.norm(jerk, axis=-1) > 2.5) * 100.0
    gen_pets = []
    for b in range(len(gen_trajs)):
        dist_matrix = np.linalg.norm(gen_trajs[b][:, None, :] - cond_trajs[b][None, :, :], axis=-1)
        min_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        gen_pets.append(abs(min_idx[0] - min_idx[1]) * dt)
    gen_pets = np.array(gen_pets)
    w1 = wasserstein_distance(real_pets, gen_pets) if len(real_pets) > 0 else 0.0
    return {"ade": ade, "fde": fde, "acc_viol": acc_viol, "jerk_viol": jerk_viol, "pet_w1": w1,
            "real_pet_mean": np.mean(real_pets), "gen_pet_mean": np.mean(gen_pets)}

def build_data():
    df = pd.read_csv(CSV)
    id_col = "event_id"
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    stats = ckpt["stats"]
    mean, std = stats["mean"], stats["std"]
    cond_mean, cond_std = stats["cond_mean"], stats["cond_std"]

    model = TrajectoryDiffusionModel(traj_shape=(TH, 1, 2), cond_dim=4, hidden_dim=128).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    gt_trajs_list, cond_list, cond_trajs_list, real_pets = [], [], [], []
    for _, grp in df.groupby(id_col):
        if "frame" in grp.columns:
            grp = grp.sort_values("frame")
        if len(grp) < TH:
            continue
        sub = grp.iloc[:TH]
        ti = sub[["x_i", "y_i"]].values.astype(np.float32)
        tj = sub[["x_j", "y_j"]].values.astype(np.float32)
        start_i = ti[0].copy()
        ti_rel = ti - start_i
        tj_rel = tj - start_i
        v_i = np.zeros_like(ti)
        v_i[1:] = np.diff(ti, axis=0)
        v_j = np.zeros_like(tj)
        v_j[1:] = np.diff(tj, axis=0)
        cond = np.hstack([(tj[0] - ti[0]), (v_j[0] - v_i[0])])
        cond_norm = (cond - cond_mean) / cond_std
        cond_list.append(cond_norm)
        gt_trajs_list.append(ti_rel)
        cond_trajs_list.append(tj_rel)
        real_pets.append(float(sub["pet"].iloc[0]) if "pet" in sub.columns else 1.5)

    gt_arr = np.array(gt_trajs_list, dtype=np.float32)
    cond_trajs_full = np.array(cond_trajs_list, dtype=np.float32)
    cond_tensor = torch.tensor(np.array(cond_list, dtype=np.float32).squeeze(1), device=DEVICE)
    real_pets_arr = np.array(real_pets, dtype=np.float32)
    return model, gt_arr, cond_trajs_full, cond_tensor, real_pets_arr

@torch.no_grad()
def sample_with_smoothing(model, cond, K=10, num_steps=50, savgol_window=5, savgol_poly=2, gaussian_sigma=1.0):
    B = cond.shape[0]
    all_gen = []
    for _ in range(K):
        x = torch.randn(B, TH, 1, 2, device=DEVICE)
        dt = 1.0 / num_steps
        for step in reversed(range(1, num_steps+1)):
            t_val = step / num_steps
            t = torch.full((B, 1), t_val, device=DEVICE)
            v_pred = model(x, cond, t)
            x = x - v_pred * dt
        gen = x.cpu().numpy().squeeze(2)  # (B, TH, 2)
        # Smooth along time axis
        try:
            gen = savgol_filter(gen, window_length=savgol_window, polyorder=savgol_poly, axis=1)
        except Exception:
            pass
        try:
            gen = gaussian_filter1d(gen, sigma=gaussian_sigma, axis=1)
        except Exception:
            pass
        all_gen.append(gen)
    all_gen = np.stack(all_gen, axis=1)  # (B, K, TH, 2)
    return all_gen

def evaluate_params(savgol_window, gaussian_sigma):
    model, gt_arr, cond_trajs_full, cond_tensor, real_pets = build_data()
    all_gen = sample_with_smoothing(model, cond_tensor, savgol_window=savgol_window, gaussian_sigma=gaussian_sigma)
    # Choose best of K by minADE
    errors = np.linalg.norm(all_gen - gt_arr[:, None, :, :], axis=-1)
    best_k = np.argmin(np.mean(errors, axis=-1), axis=1)
    best = all_gen[np.arange(len(all_gen)), best_k]
    return compute_metrics(best, gt_arr, cond_trajs_full, real_pets)

if __name__ == "__main__":
    print(f"{'savgol_window':>12} {'gauss_sigma':>12} {'ADE':>8} {'FDE':>8} {'Acc%':>6} {'Jerk%':>7} {'PET W1':>7}")
    for sw in [5, 7]:
        for gs in [0.5, 0.8, 1.0, 1.2, 1.5]:
            r = evaluate_params(sw, gs)
            print(f"{sw:>12} {gs:>12} {r['ade']:>8.4f} {r['fde']:>8.4f} {r['acc_viol']:>6.2f} {r['jerk_viol']:>7.2f} {r['pet_w1']:>7.4f}")
