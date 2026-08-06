import os
import sys

# Automatically prepend parent root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from traffic_diffusion.trajectory_diffusion import TrajectoryDiffusionModel

def load_test_tensors(test_csv_path, mean, std, Th=16):
    df = pd.read_csv(test_csv_path)
    
    # Dynamic Event ID Column Detection
    id_candidates = ["conflict_id", "event_id", "pair_id", "track_id", "case_id", "id", "event"]
    id_col = next((c for c in id_candidates if c in df.columns), None)
    
    if id_col is None:
        print(f"⚠️ No explicit event ID column found in {test_csv_path}. Chunking into sequences of length {Th}.")
        df["conflict_id"] = np.arange(len(df)) // Th
        id_col = "conflict_id"

    cond_trajs = []
    gt_trajs_list = []
    start_pos_list = []
    real_pets = []
    
    for _, grp in df.groupby(id_col):
        if "frame" in grp.columns:
            grp = grp.sort_values("frame")
            
        if len(grp) < Th:
            continue
        sub = grp.iloc[:Th]
        
        ti = sub[["x_i", "y_i"]].values.astype(np.float32)
        tj = sub[["x_j", "y_j"]].values.astype(np.float32)
        
        start_i = ti[0].copy()
        ti_rel = ti - start_i
        
        cond_trajs.append(tj[0] - start_i)
        gt_trajs_list.append(ti_rel)
        start_pos_list.append(start_i)
        
        if "pet" in sub.columns:
            real_pets.append(float(sub["pet"].iloc[0]))
        elif "PET" in sub.columns:
            real_pets.append(float(sub["PET"].iloc[0]))
        else:
            real_pets.append(1.5)

    gt_arr = np.array(gt_trajs_list, dtype=np.float32)
    cond_arr = np.array(cond_trajs, dtype=np.float32)
    start_pos_arr = np.array(start_pos_list, dtype=np.float32)
    real_pets_arr = np.array(real_pets, dtype=np.float32)

    gt_norm = (gt_arr - mean) / std
    cond_norm = (cond_arr - mean) / std
    
    cond_tensor = torch.tensor(cond_norm, dtype=torch.float32)
    return cond_tensor, gt_arr, cond_arr, start_pos_arr, real_pets_arr

def compute_metrics(gen_trajs, gt_trajs, cond_trajs, real_pets, dt=0.1):
    # ADE / FDE
    errors = np.linalg.norm(gen_trajs - gt_trajs, axis=-1)
    ade = np.mean(errors)
    fde = np.mean(errors[:, -1])
    
    # Kinematics
    vel = np.diff(gen_trajs, axis=1) / dt
    acc = np.diff(vel, axis=1) / dt
    jerk = np.diff(acc, axis=1) / dt
    
    acc_mag = np.linalg.norm(acc, axis=-1)
    jerk_mag = np.linalg.norm(jerk, axis=-1)
    
    acc_viol = np.mean(acc_mag > 4.5) * 100.0 if acc_mag.size > 0 else 0.0
    jerk_viol = np.mean(jerk_mag > 2.5) * 100.0 if jerk_mag.size > 0 else 0.0
    
    # Post-Encroachment Time (PET) calculation
    gen_pets = []
    for b in range(len(gen_trajs)):
        dist_matrix = np.linalg.norm(gen_trajs[b][:, None, :] - cond_trajs[b][None, :, :], axis=-1)
        min_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        pet_val = abs(min_idx[0] - min_idx[1]) * dt
        gen_pets.append(pet_val)
    gen_pets = np.array(gen_pets)
    
    w1_dist = wasserstein_distance(real_pets, gen_pets) if len(real_pets) > 0 else 0.0
    
    return {
        'ade': ade,
        'fde': fde,
        'acc_violations': acc_viol,
        'jerk_violations': jerk_viol,
        'pet_w1': w1_dist,
        'real_pet_mean': np.mean(real_pets),
        'real_pet_std': np.std(real_pets),
        'gen_pet_mean': np.mean(gen_pets),
        'gen_pet_std': np.std(gen_pets)
    }

def run_evaluation():
    print("=" * 65)
    print("🚀 EVALUATING TEMPORAL U-NET DIFFUSION MODEL ON HELD-OUT TEST SET")
    print("=" * 65)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = "checkpoints/traj_diffusion_best.pt"
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found at {ckpt_path}")
        
    checkpoint = torch.load(ckpt_path, map_location=device)
    mean = checkpoint["mean"]
    std = checkpoint["std"]
    
    test_csv = "outputs/petevents_test.csv"
    cond_tensor, gt_trajs, cond_trajs, start_positions, real_pets = load_test_tensors(test_csv, mean, std, Th=16)
    cond_tensor = cond_tensor.to(device)
    
    model = TrajectoryDiffusionModel(traj_shape=(16, 1, 2), cond_dim=2, hidden_dim=128).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    
    K = 10
    all_generated = []
    print(f"🔄 Generating {K} samples per test case...")
    
    with torch.no_grad():
        for k in range(K):
            gen_samples = model.sample(cond_tensor)
            gen_np = gen_samples.cpu().squeeze(2).numpy()
            
            # Reconstruct relative trajectory and denormalize
            gen_rel = (gen_np * std) + mean
            all_generated.append(gen_rel)
            
    pred_k_trajs = np.stack(all_generated, axis=1) # Shape: (N, K, 16, 2)
    
    # Best-of-K selection based on minimum displacement error
    errors = np.linalg.norm(pred_k_trajs - gt_trajs[:, None, :, :], axis=-1)
    ade_per_k = np.mean(errors, axis=-1)
    fde_per_k = errors[:, :, -1]
    best_k_idx = np.argmin(ade_per_k + fde_per_k, axis=1)
    
    best_trajs = pred_k_trajs[np.arange(len(pred_k_trajs)), best_k_idx]
    
    results = compute_metrics(best_trajs, gt_trajs, cond_trajs, real_pets)
    
    print("\n" + "=" * 65)
    print("📋 HELD-OUT TEST SET BENCHMARK RESULTS")
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
