
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    from traffic_diffusion.trajectory_diffusion import load_trajdiff_dataset

    raw_dataset, meta_df = load_trajdiff_dataset(ROOT)

    train_loader, eval_loader, train_dataset, eval_dataset, stats = \
        build_clean_dataloaders(raw_dataset, batch_size=14, T=15, N=1, F=4)

    model = create_model(device, T=15, N=1, F=4, cond_dim=4)
    best_ckpt, last_ckpt, history = train_diffusion_model(
        model, train_loader, device,
        num_epochs=50,
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
