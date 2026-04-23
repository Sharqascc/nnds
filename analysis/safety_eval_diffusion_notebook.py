
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from traffic_diffusion.training_utils import build_clean_dataloaders, create_model, train_diffusion_model
from traffic_diffusion.sampling_utils import load_eval_model, sample_future
from traffic_diffusion.pet_safety_metrics import compute_safety_metrics

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

src_path = "/content/nnds/outputs/petevents_20_frames_test.csv"




def load_trajdiff_dataset(root: Path, src_path: str):
    """Load PET events CSV into a raw_dataset suitable for build_clean_dataloaders.

    raw_dataset is a list of (trajectory_tensor_flat, condition_tensor).
    All trajectories are padded/truncated to fixed length T_TARGET.
    """
    import ast
    import torch
    import pandas as pd

    T_TARGET = 15  # must match T used in build_clean_dataloaders/create_model
    N = 1
    F = 4

    df = pd.read_csv(src_path)
    raw_dataset = []

    for _, row in df.iterrows():
        traj_i = ast.literal_eval(row["world_traj_i"])
        # traj_i is list of (t, x, y)
        if len(traj_i) == 0:
            continue

        # Build trajectory tensor (T_TARGET, 1, 4)
        traj_tensor = torch.zeros(T_TARGET, N, F, dtype=torch.float32)

        # Fill with as many steps as we have, up to T_TARGET
        for t_idx, (t, x, y) in enumerate(traj_i[:T_TARGET]):
            traj_tensor[t_idx, 0, 0] = float(t)
            traj_tensor[t_idx, 0, 1] = float(x)
            traj_tensor[t_idx, 0, 2] = float(y)
            traj_tensor[t_idx, 0, 3] = 1.0  # dummy feature

        # Condition: start & end positions of actor i (from available steps)
        t0, x0, y0 = traj_i[0]
        t1, x1, y1 = traj_i[-1]
        cond_tensor = torch.tensor(
            [x0, y0, x1, y1],
            dtype=torch.float32
        )

        # Flatten trajectory to (T*N*F,) so all have same length
        traj_flat = traj_tensor.view(-1)
        raw_dataset.append((traj_flat, cond_tensor))

    meta_df = df
    return raw_dataset, meta_df

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    raw_dataset, meta_df = load_trajdiff_dataset(ROOT, src_path)

    train_loader, eval_loader, train_dataset, eval_dataset, stats = \
        build_clean_dataloaders(raw_dataset, batch_size=14, T=15, N=1, F=4)

    model = create_model(device, T=15, N=1, F=4, cond_dim=4)
    best_ckpt, last_ckpt, history = train_diffusion_model(
        model, train_loader, device,
        num_epochs=1,
        save_dir=str(ROOT / "checkpoints"),
    )

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(OUTPUT_DIR / "training_history.csv", index=False)

    eval_model = load_eval_model(best_ckpt, device, T=15, N=1, F=4, cond_dim=4)
    samples = sample_future(eval_model, eval_loader, device, T=15, N=1, F=4,
                            num_samples=10, num_steps=100)

    if samples is None:
        raise RuntimeError("No samples produced.")

    S, B_total, D = samples.shape
    df_samples = pd.DataFrame(samples[0])
    df_samples.to_csv(OUTPUT_DIR / "safety_eval_diffusion_simple.csv", index=False)

    eval_indices = eval_dataset.indices if hasattr(eval_dataset, "indices") else eval_dataset.dataset.indices
    meta_eval = meta_df.iloc[eval_indices].reset_index(drop=True)

    B_eval = df_samples.shape[0]
    traj_pred = df_samples.to_numpy(dtype=np.float32).reshape(B_eval, 15, 1, 4)

    records = []
    for i in range(B_eval):
        meta_i = meta_eval.iloc[i]
        pet_model = float(meta_i.get("PET", np.nan))
        if np.isfinite(pet_model):
            risk_model = float(np.clip(1.0 - pet_model, 0.0, 1.0))
        else:
            risk_model = 0.0

        records.append({
            "idx": int(i),
            "obj_i": meta_i.get("obj_i", None),
            "obj_j": meta_i.get("obj_j", None),
            "cell_id": meta_i.get("cell_id", None),
            "PET": pet_model,
            "risk_score": risk_model,
            "severity_evt": (
                "critical" if pet_model < 0.5
                else "high" if pet_model < 1.0
                else "medium"
            ) if np.isfinite(pet_model) else None,
        })

    df_events_model = pd.DataFrame(records)
    df_events_model.to_csv(OUTPUT_DIR / "safety_events_diffusion_model.csv", index=False)

    summary_model = compute_safety_metrics(df_events_model)
    df_summary_model = pd.DataFrame([summary_model])
    df_summary_model.to_csv(OUTPUT_DIR / "safety_eval_diffusion_summary.csv", index=False)

if __name__ == "__main__":
    main()
