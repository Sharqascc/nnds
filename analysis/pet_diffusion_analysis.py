
import ast
import numpy as np
import pandas as pd
import torch

def parse_traj(cell):
    try:
        return np.array(ast.literal_eval(cell), dtype=float)
    except Exception:
        return None

def compute_pet_like_metrics(batch, sample_future_fn, scale, noise_scale=0.01, d_thresh=1.0):
    B = batch["past"].shape[0]

    past_norm   = batch["past"]
    future_norm = batch["future"]

    past_world   = past_norm * scale
    real_world   = future_norm * scale
    sample_world = sample_future_fn(batch, max_B=B, noise_scale=noise_scale)

    real_np   = real_world.cpu().numpy()
    sample_np = sample_world.cpu().numpy()

    pet_pairs = []

    for b in range(B):
        real_traj   = real_np[b]
        sample_traj = sample_np[b]

        real_dist   = np.linalg.norm(real_traj[:, 0:2]   - real_traj[:, 2:4],   axis=-1)
        sample_dist = np.linalg.norm(sample_traj[:, 0:2] - sample_traj[:, 2:4], axis=-1)

        def first_hit(dist):
            idxs = np.where(dist < d_thresh)[0]
            if len(idxs) == 0:
                return None
            return float(idxs[0])

        pet_real   = first_hit(real_dist)
        pet_sample = first_hit(sample_dist)

        pet_pairs.append((pet_real, pet_sample))

    reals = [p[0] for p in pet_pairs if p[0] is not None]
    samps = [p[1] for p in pet_pairs if p[1] is not None]
    both  = [p for p in pet_pairs if p[0] is not None and p[1] is not None]

    print("=== PET-like metrics (per example, in future steps) ===")
    print("d_thresh:", d_thresh)
    print(f"num examples in batch: {len(pet_pairs)}")
    print(f"real PET defined in:   {len(reals)} examples")
    print(f"sample PET defined in: {len(samps)} examples")
    print(f"both defined in:       {len(both)} examples")

    if both:
        diffs = [p[1] - p[0] for p in both]
        print(f"mean(sample PET - real PET): {np.mean(diffs):.3f}")
        print(f"std(sample PET - real PET):  {np.std(diffs):.3f}")

    print("\nExample PET pairs (real, sample) for first 5 examples:")
    for i, (pr, ps) in enumerate(pet_pairs[:5]):
        print(f"  idx {i}: real={pr}, sample={ps}")

    return pet_pairs


def compare_realPET_samplePET(df_pet_path, batch, sample_future_fn, scale,
                              noise_scale=0.01, d_thresh=1.0):
    df_pet = pd.read_csv(df_pet_path)
    print("df_pet rows:", len(df_pet))

    idxs = batch["idx"].cpu().numpy()
    B = len(idxs)

    future_norm = batch["future"]
    real_world   = future_norm * scale
    sample_world = sample_future_fn(batch, max_B=B, noise_scale=noise_scale)

    real_np   = real_world.cpu().numpy()
    sample_np = sample_world.cpu().numpy()

    records = []

    for b in range(B):
        row_idx = int(idxs[b])
        true_pet = float(df_pet.loc[row_idx, "PET"])

        real_traj   = real_np[b]
        sample_traj = sample_np[b]

        real_dist   = np.linalg.norm(real_traj[:, 0:2]   - real_traj[:, 2:4],   axis=-1)
        sample_dist = np.linalg.norm(sample_traj[:, 0:2] - sample_traj[:, 2:4], axis=-1)

        def first_hit(dist):
            idxs_local = np.where(dist < d_thresh)[0]
            if len(idxs_local) == 0:
                return None
            return float(idxs_local[0])

        pet_like_real   = first_hit(real_dist)
        pet_like_sample = first_hit(sample_dist)

        records.append((row_idx, true_pet, pet_like_real, pet_like_sample))

    print("=== Real PET (CSV) vs PET-like (steps) from real vs sample ===")
    print(f"d_thresh: {d_thresh}")
    print(f"batch size: {B}")
    both = [r for r in records if r[3] is not None]
    if both:
        diffs = [r[3] - r[2] for r in both if r[2] is not None]
        if diffs:
            print(f"sample-real PET-like step diff (where both defined): "
                  f"mean={np.mean(diffs):.3f}, std={np.std(diffs):.3f}")
    else:
        print("No examples where sample PET-like is defined.")

    print("\nFirst 5 examples:")
    print("row_idx | true PET (s) | PET-like real (steps) | PET-like sample (steps)")
    for r in records[:5]:
        print(f"{r[0]:7d} | {r[1]:12.6f} | {str(r[2]):20} | {str(r[3]):22}")

    return records
