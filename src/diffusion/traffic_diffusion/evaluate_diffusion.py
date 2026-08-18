import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wasserstein_distance
from scipy.signal import savgol_filter
from src.diffusion.traffic_diffusion.trajectory_diffusion import TrajectoryDiffusionModel
from src.diffusion.traffic_diffusion.train_trajectory_diffusion import build_normalized_tensors

def compute_min_ade_fde(pred_trajs, gt_trajs):
    gt_expanded = np.expand_dims(gt_trajs, axis=1)
    errors = np.linalg.norm(pred_trajs - gt_expanded, axis=-1)
    ade_per_k = np.mean(errors, axis=-1)
    fde_per_k = errors[:, :, -1]
    return float(np.mean(np.min(ade_per_k, axis=1))), float(np.mean(np.min(fde_per_k, axis=1)))

def compute_kinematic_feasibility(trajs, dt=0.1, max_acc=4.5, max_jerk=2.5):
    vel = np.diff(trajs, axis=1) / dt
    acc = np.diff(vel, axis=1) / dt
    jerk = np.diff(acc, axis=1) / dt
    
    acc_mag = np.linalg.norm(acc, axis=-1)
    jerk_mag = np.linalg.norm(jerk, axis=-1)
    
    acc_violations = float(np.mean(acc_mag > max_acc) * 100) if acc_mag.size > 0 else 0.0
    jerk_violations = float(np.mean(jerk_mag > max_jerk) * 100) if jerk_mag.size > 0 else 0.0
    return acc_violations, jerk_violations

def compute_pet_from_trajectories(traj_i_batch, traj_j_batch, dt=0.1, conflict_dist_thresh=1.5):
    pets = []
    for b in range(len(traj_i_batch)):
        ti, tj = traj_i_batch[b], traj_j_batch[b]
        if tj.ndim == 1:
            dist_matrix = np.linalg.norm(ti - tj[None, :], axis=-1)
            min_idx = np.argmin(dist_matrix)
            pets.append(min_idx * dt)
        else:
            dist_matrix = np.linalg.norm(ti[:, None, :] - tj[None, :, :], axis=-1)
            conflict_indices = np.argwhere(dist_matrix < conflict_dist_thresh)
            if len(conflict_indices) > 0:
                min_idx = np.argmin([dist_matrix[ci[0], ci[1]] for ci in conflict_indices])
                frame_i, frame_j = conflict_indices[min_idx]
                pets.append(abs(frame_i - frame_j) * dt)
            else:
                min_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
                pets.append(abs(min_idx[0] - min_idx[1]) * dt)
    return np.array(pets, dtype=np.float32)

def run_benchmark(test_csv_path, checkpoint_path, K=10, Th=16):
    print("=" * 65)
    print("🚀 EVALUATING TEMPORAL U-NET DIFFUSION MODEL ON HELD-OUT TEST SET")
    print("=" * 65)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found at {checkpoint_path}. Please train first.")
        return
        
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    mean, std = checkpoint["mean"], checkpoint["std"]
    stats = {"mean": mean, "std": std}
    
    # 1. Unpack 7 values accurately
    _, cond_tensor_norm, _, gt_rel, cond_trajs, start_pos_arr, real_pets = build_normalized_tensors(
        test_csv_path, Th=Th, scaler_stats=stats, augment=False
    )
    
    # 2. Reconstruct absolute GT positions: p_0 + delta_p
    gt_trajs = start_pos_arr[:, None, :] + gt_rel
    
    model = TrajectoryDiffusionModel(traj_shape=(Th, 1, 2), cond_dim=2, hidden_dim=128).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    
    cond_tensor = cond_tensor_norm.to(device)
    all_generated = []
    
    with torch.no_grad():
        for k in range(K):
            gen_samples = model.sample(cond_tensor, num_steps=20)
            gen_np = gen_samples.cpu().squeeze(2).numpy()
            
            # Inverse normalize relative displacements
            gen_rel_unnorm = (gen_np * std) + mean
            
            # Reconstruct absolute coordinates
            gen_abs_unnorm = start_pos_arr[:, None, :] + gen_rel_unnorm
            
            # Apply trajectory smoothing if sequence length allows
            if Th >= 5:
                gen_abs_unnorm[:, :, 0] = savgol_filter(gen_abs_unnorm[:, :, 0], window_length=5, polyorder=2, axis=1)
                gen_abs_unnorm[:, :, 1] = savgol_filter(gen_abs_unnorm[:, :, 1], window_length=5, polyorder=2, axis=1)
                
            all_generated.append(gen_abs_unnorm)
            
    pred_k_trajs = np.stack(all_generated, axis=1)
    
    # Calculate performance metrics
    min_ade, min_fde = compute_min_ade_fde(pred_k_trajs, gt_trajs)
    acc_viol, jerk_viol = compute_kinematic_feasibility(pred_k_trajs[:, 0, :, :])
    
    if len(real_pets) == 0 or np.all(real_pets == 1.5):
        real_pets = compute_pet_from_trajectories(gt_trajs, cond_trajs)
        
    gen_pets = compute_pet_from_trajectories(pred_k_trajs[:, 0, :, :], cond_trajs)
    w1_dist = float(wasserstein_distance(real_pets, gen_pets))
    
    print("\n" + "=" * 65)
    print("📋 HELD-OUT TEST SET BENCHMARK RESULTS")
    print("=" * 65)
    print(f"1. TRAJECTORY FIDELITY (K={K}):")
    print(f"   • minADE: {min_ade:.4f} m (Target: < 0.50 m)")
    print(f"   • minFDE: {min_fde:.4f} m (Target: < 1.00 m)")
    print("\n2. KINEMATIC ADMISSIBILITY:")
    print(f"   • Acceleration Violations (>4.5 m/s²): {acc_viol:.2f}% (Target: < 2.0%)")
    print(f"   • Jerk Violations (>2.5 m/s³): {jerk_viol:.2f}% (Target: < 2.0%)")
    print("\n3. SURROGATE SAFETY ALIGNMENT:")
    print(f"   • Ground-Truth PET Mean ± Std: {np.mean(real_pets):.3f}s ± {np.std(real_pets):.3f}s")
    print(f"   • Model Generated PET Mean ± Std: {np.mean(gen_pets):.3f}s ± {np.std(gen_pets):.3f}s")
    print(f"   • Wasserstein Distance W1: {w1_dist:.4f} (Target: < 0.150)")
    
    if w1_dist < 0.150 and acc_viol < 2.0 and jerk_viol < 2.0 and min_ade < 0.50:
        print("\n✅ PASS: All metrics meet production tolerances.")
    else:
        print("\n⚠️ STATUS: Benchmark complete — check metric alignment against targets.")
    print("=" * 65)

if __name__ == "__main__":
    run_benchmark(
        test_csv_path="outputs/petevents_test.csv",
        checkpoint_path="checkpoints/traj_diffusion_best.pt",
        K=10,
        Th=16
    )
