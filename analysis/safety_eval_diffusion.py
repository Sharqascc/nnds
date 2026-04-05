import os
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader

# -----------------------------
# 1) Dataset from petevents_bev.csv
# -----------------------------

def parse_traj_txy(cell):
    arr = np.array(eval(cell), dtype=float)   # (T,3): t,x,y
    return arr[:, 0], arr[:, 1:3]

class TrajDiffusionDatasetNorm(Dataset):
    def __init__(self, csv_path, Th=8):
        df = pd.read_csv(csv_path)

        t_i_list, xy_i_list, xy_j_list = [], [], []
        lengths = []

        for ci, cj in zip(df["world_traj_i"], df["world_traj_j"]):
            t_i, xy_i = parse_traj_txy(ci)
            t_j, xy_j = parse_traj_txy(cj)
            T = min(len(t_i), len(t_j))
            if T < 2:
                continue
            lengths.append(T)
            t_i_list.append(t_i[:T])
            xy_i_list.append(xy_i[:T])
            xy_j_list.append(xy_j[:T])

        if not lengths:
            raise RuntimeError("No valid trajectories found in CSV")

        T_max = max(lengths)
        traj_list = []

        for xy_i, xy_j in zip(xy_i_list, xy_j_list):
            T = min(len(xy_i), T_max)
            pad_len = T_max - T
            xy_i_p = np.pad(xy_i[:T], ((0, pad_len), (0, 0)), mode="edge")
            xy_j_p = np.pad(xy_j[:T], ((0, pad_len), (0, 0)), mode="edge")
            traj_list.append(np.concatenate([xy_i_p, xy_j_p], axis=-1))

        traj_all = np.stack(traj_list, axis=0)   # (N, T_max, 4)

        N, T, D = traj_all.shape
        Th = min(Th, T // 2)
        inputs_np = traj_all[:, :Th, :]
        future_np = traj_all[:, Th:, :]

        centers = inputs_np[:, -1, 0:2]

        inputs_norm = inputs_np.copy()
        targets_norm = future_np.copy()

        for k in range(N):
            cx, cy = centers[k]
            inputs_norm[k, :, 0:4] -= np.array([cx, cy, cx, cy])
            targets_norm[k, :, 0:4] -= np.array([cx, cy, cx, cy])

        scale = 1.0
        inputs_norm /= scale
        targets_norm /= scale

        self.inputs = torch.from_numpy(inputs_norm).float()
        self.targets = torch.from_numpy(targets_norm).float()
        self.centers = torch.from_numpy(centers).float()
        self.scale = scale
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return {
            "idx": idx,
            "past": self.inputs[idx],
            "future": self.targets[idx],
            "center": self.centers[idx],
        }

def make_loader(csv_path, batch_size=32, Th=8, shuffle=False):
    ds = TrajDiffusionDatasetNorm(csv_path, Th=Th)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    return ds, dl

# -----------------------------
# 2) PET / TTC helpers
# -----------------------------

def first_below_threshold(dist_seq, thresh):
    idxs = np.where(dist_seq < thresh)[0]
    if len(idxs) == 0:
        return None
    return float(idxs[0])

def compute_ttc_seq(pos1, pos2, dt, d_thresh=1.0, eps=1e-6):
    v1 = (pos1[1:] - pos1[:-1]) / dt
    v2 = (pos2[1:] - pos2[:-1]) / dt
    v_rel = v2 - v1
    p_rel = pos2[:-1] - pos1[:-1]

    ttc = []
    for p, v in zip(p_rel, v_rel):
        vr2 = float(np.dot(v, v))
        if vr2 < eps:
            ttc.append(None)
            continue
        t_star = -float(np.dot(p, v)) / vr2
        if t_star <= 0:
            ttc.append(None)
            continue
        d_min = float(np.linalg.norm(p + t_star * v))
        if d_min < d_thresh:
            ttc.append(t_star)
        else:
            ttc.append(None)
    return ttc

@torch.no_grad()
def eval_safety_over_loader(
    loader,
    df_pet,
    sample_future_fn,
    scale=1.0,
    out_csv_path="outputs/safety_eval_diffusion.csv",
    num_samples=20,
    d_thresh=1.0,
    dt=0.0333,
    max_batches=None,
    device="cuda",
):
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)

    all_records = []

    for b_idx, batch in enumerate(loader):
        if max_batches is not None and b_idx >= max_batches:
            break

        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)

        B = batch["future"].shape[0]
        future_world = batch["future"] * scale
        future_np = future_world.cpu().numpy()

        samples_np = sample_future_fn(batch, num_samples=num_samples)  # (S,B,T,D)

        for i in range(B):
            row_idx = int(batch["idx"][i].item())

            true_pet = None
            if "PET" in df_pet.columns and row_idx < len(df_pet):
                true_pet = float(df_pet.loc[row_idx, "PET"])

            real_traj = future_np[i]
            p1_real = real_traj[:, 0:2]
            p2_real = real_traj[:, 2:4]

            dist_real = np.linalg.norm(p1_real - p2_real, axis=-1)
            pet_like_real = first_below_threshold(dist_real, d_thresh)

            ttc_seq_real = compute_ttc_seq(p1_real, p2_real, dt, d_thresh=d_thresh)
            real_ttc_vals = [t for t in ttc_seq_real if t is not None]
            min_ttc_real = min(real_ttc_vals) if real_ttc_vals else None

            pet_samples = []
            ttc_samples = []

            for s in range(num_samples):
                sample_traj = samples_np[s, i]
                p1_s = sample_traj[:, 0:2]
                p2_s = sample_traj[:, 2:4]

                dist_s = np.linalg.norm(p1_s - p2_s, axis=-1)
                pet_s = first_below_threshold(dist_s, d_thresh)
                pet_samples.append(pet_s)

                ttc_seq_s = compute_ttc_seq(p1_s, p2_s, dt, d_thresh=d_thresh)
                ttc_vals_s = [t for t in ttc_seq_s if t is not None]
                ttc_s = min(ttc_vals_s) if ttc_vals_s else None
                ttc_samples.append(ttc_s)

            def finite_vals(xs):
                return [x for x in xs if x is not None]

            pet_s_f = finite_vals(pet_samples)
            ttc_s_f = finite_vals(ttc_samples)

            rec = {
                "idx": row_idx,
                "true_PET": true_pet,
                "real_pet_like_step": pet_like_real,
                "real_min_TTC": min_ttc_real,
                "sample_pet_like_step_mean": float(np.mean(pet_s_f)) if pet_s_f else None,
                "sample_pet_like_step_std": float(np.std(pet_s_f)) if pet_s_f else None,
                "sample_min_TTC_mean": float(np.mean(ttc_s_f)) if ttc_s_f else None,
                "sample_min_TTC_std": float(np.std(ttc_s_f)) if ttc_s_f else None,
                "sample_pet_defined_frac": len(pet_s_f) / num_samples,
                "sample_ttc_defined_frac": len(ttc_s_f) / num_samples,
            }
            all_records.append(rec)

        print(f"Processed batch {b_idx+1}, total records: {len(all_records)}")

    df_out = pd.DataFrame.from_records(all_records)
    df_out.to_csv(out_csv_path, index=False)
    print("Saved safety eval to:", out_csv_path)

# -----------------------------
# 3) Entry point (wire your model here)
# -----------------------------

def sample_future_fn_stub(batch, num_samples=20):
    raise NotImplementedError("Plug in your sample_future_denorm-based function here.")

def main():
    csv_path = "outputs/petevents_bev.csv"
    ds, dl = make_loader(csv_path, batch_size=32, Th=8, shuffle=False)
    df_pet = ds.df

    # TODO: replace this stub with your real sampler
    def sample_fn(batch, num_samples=20):
        return sample_future_fn_stub(batch, num_samples=num_samples)

    eval_safety_over_loader(
        loader=dl,
        df_pet=df_pet,
        sample_future_fn=sample_fn,
        scale=ds.scale,
        out_csv_path="outputs/safety_eval_diffusion.csv",
        num_samples=20,
        d_thresh=1.0,
        dt=0.0333,
        max_batches=None,
        device="cuda",
    )

if __name__ == "__main__":
    main()
