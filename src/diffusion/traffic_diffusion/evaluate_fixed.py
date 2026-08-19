import os
import torch
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from src.diffusion.traffic_diffusion.trajectory_diffusion import TrajectoryDiffusionModel

def load_checkpoint_safe(ckpt_path, device):
    try:
        return torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(ckpt_path, map_location=device)

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
    
    w1_dist = wasserstein_distance(real_pets, gen_pets) if len(real_pets) > 0 else 0.0
    
    return {
        'ade': ade, 'fde': fde, 'acc_violations': acc_viol, 
        'jerk_violations': jerk_viol, 'pet_w1': w1_dist,
        'real_pet_mean': np.mean(real_pets), 'real_pet_std': np.std(real_pets),
        'gen_pet_mean': np.mean(gen_pets), 'gen_pet_std': np.std(gen_pets)
    }

def run_evaluation(test_csv_path="outputs/petevents_test.csv", ckpt_path="checkpoints/traj_diffusion_best.pt", Th=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n=========================================================")
    print("🚀 EVALUATING FLOW MATCHING VELOCITY MODEL ON TEST SET")
    print("=========================================================")
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found at {ckpt_path}")
        
    checkpoint = load_checkpoint_safe(ckpt_path, device)
    stats = checkpoint["stats"]
    mean = stats["mean"]
    std = stats["std"]
    cond_mean = stats["cond_mean"]
    cond_std = stats["cond_std"]
    
    model = TrajectoryDiffusionModel(traj_shape=(Th, 1, 2), cond_dim=4, hidden_dim=128).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    
    df_test = pd.read_csv(test_csv_path)
    id_col = next((c for c in ["conflict_id", "event_id", "pair_id", "track_id", "id"] if c in df_test.columns), None)
    
    if id_col is None:
        df_test["conflict_id"] = np.arange(len(df_test)) // Th
        id_col = "conflict_id"

    gt_trajs_list = []
    cond_trajs_list = []
    cond_list = []
    real_pets = []
    
    with torch.no_grad():
        for _, grp in df_test.groupby(id_col):
            if "frame" in grp.columns:
                grp = grp.sort_values("frame")
            if len(grp) < Th:
                continue
            sub = grp.iloc[:Th]
            
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
            
            if "pet" in sub.columns:
                real_pets.append(float(sub["pet"].iloc[0]))
            else:
                real_pets.append(1.5)

    gt_arr = np.array(gt_trajs_list, dtype=np.float32)
    cond_trajs_full = np.array(cond_trajs_list, dtype=np.float32)
    cond_tensor = torch.tensor(np.array(cond_list, dtype=np.float32).squeeze(1), device=device)
    real_pets_arr = np.array(real_pets, dtype=np.float32)

    K = 10
    all_generated_pos = []
    
    with torch.no_grad():
        for k in range(K):
            v_sampled_norm = model.sample(cond=cond_tensor).cpu().numpy()
            
            if v_sampled_norm.ndim == 4:
                v_sampled_norm = v_sampled_norm.squeeze(2)
                
            v_sampled = (v_sampled_norm * std) + mean
            if v_sampled.ndim == 4:
                v_sampled = v_sampled.squeeze(2)
            
            # Reconstruct absolute positions via cumulative summation of velocities
            gen_pos = np.cumsum(v_sampled, axis=1)
            all_generated_pos.append(gen_pos)
            
    pred_k_trajs = np.stack(all_generated_pos, axis=1) # (N, K, Th, 2)
    
    errors = np.linalg.norm(pred_k_trajs - gt_arr[:, None, :, :], axis=-1)
    best_k_idx = np.argmin(np.mean(errors, axis=-1) + errors[:, :, -1], axis=1)
    best_trajs = pred_k_trajs[np.arange(len(pred_k_trajs)), best_k_idx]
    
    results = compute_metrics(best_trajs, gt_arr, cond_trajs_full, real_pets_arr)
    
    print("\n" + "=" * 65)
    print("📋 BENCHMARK RESULTS (FLOW MATCHING VELOCITY MODEL)")
    print("=" * 65)
    print(f"1. TRAJECTORY FIDELITY (K={K}):")
    print(f"   • minADE: {results['ade']:.4f} m (Target: < 0.50 m)")
    print(f"   • minFDE: {results['fde']:.4f} m (Target: < 1.00 m)")
    print("\n2. KINEMATIC ADMISSIBILITY:")
    print(f"   • Acceleration Violations: {results['acc_violations']:.2f}% (Target: < 2.0%)")
    print(f"   • Jerk Violations: {results['jerk_violations']:.2f}% (Target: < 2.0%)")
    print("\n3. SURROGATE SAFETY ALIGNMENT:")
    print(f"   • Ground-Truth PET: {results['real_pet_mean']:.3f}s ± {results['real_pet_std']:.3f}s")
    print(f"   • Generated PET:    {results['gen_pet_mean']:.3f}s ± {results['gen_pet_std']:.3f}s")
    print(f"   • Wasserstein Distance (W1): {results['pet_w1']:.4f} (Target: < 0.150)")
    print("=" * 65)

if __name__ == "__main__":
    run_evaluation()
