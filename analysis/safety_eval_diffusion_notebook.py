"""
safety_eval_diffusion_notebook.py

Notebook-friendly pipeline for:
- Training a trajectory diffusion model on NNDS data
- Sampling counterfactual futures
- Computing PET-based safety metrics
- Exporting CSVs for downstream analysis

Intended for exploratory work and light CLI use.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch

from traffic_diffusion.training_utils import (
    build_clean_dataloaders,
    create_model,
    train_diffusion_model,
)
from traffic_diffusion.sampling_utils import load_eval_model, sample_future
from traffic_diffusion.pet_safety_metrics import compute_safety_metrics

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Notebook-friendly diffusion safety evaluation pipeline"
    )
    parser.add_argument("--T", type=int, default=15, help="Number of time steps")
    parser.add_argument("--batch-size", type=int, default=14, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=50, help="Training epochs")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of diffusion samples per batch element",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Number of diffusion steps (sampling iterations)",
    )
    parser.add_argument(
        "--risk-half-life",
        type=float,
        default=1.0,
        help="Exponential PET half-life in seconds for risk mapping",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing CSV outputs without prompting",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def safe_save(df: pd.DataFrame, path: Path, force: bool = False) -> None:
    """Save DataFrame with overwrite protection."""
    if path.exists() and not force:
        response = input(f"{path} exists. Overwrite? [y/N] ")
        if response.lower() != "y":
            logger.info("Skipped saving to %s", path)
            return
    df.to_csv(path, index=False)
    logger.info("Saved to %s", path)


def pet_to_risk_exponential(pet: float, half_life: float) -> float:
    """
    Map PET (seconds) to a [0,1] risk score using exponential decay.

    risk = exp(-pet / half_life)

    This is still a heuristic, but it is at least monotone and decays smoothly.
    """
    if not np.isfinite(pet) or pet < 0:
        return 0.0
    return float(np.exp(-pet / max(half_life, 1e-6)))


def run_safety_eval_pipeline(
    T: int,
    batch_size: int,
    num_epochs: int,
    num_samples: int,
    num_steps: int,
    risk_half_life: float,
    force: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run training + sampling + safety evaluation for the trajectory diffusion model.

    Parameters
    ----------
    T : int
        Number of time steps.
    batch_size : int
        Training batch size.
    num_epochs : int
        Number of training epochs.
    num_samples : int
        Number of diffusion samples per batch element.
    num_steps : int
        Number of diffusion sampling steps.
    risk_half_life : float
        PET half-life (seconds) for exponential risk mapping.
    force : bool
        If True, overwrite existing CSV outputs without prompting.

    Returns
    -------
    hist_df : pd.DataFrame
        Training history.
    df_events_model : pd.DataFrame
        Event-level safety records.
    df_summary_model : pd.DataFrame
        Aggregated safety metrics.
    """
    from traffic_diffusion.trajectory_diffusion import load_trajdiff_dataset

    logger.info("Loading trajectory diffusion dataset from %s", ROOT)
    raw_dataset, meta_df = load_trajdiff_dataset(ROOT)

    N = 1
    F = 4

    logger.info("Building dataloaders (batch_size=%d, T=%d, N=%d, F=%d)", batch_size, T, N, F)
    train_loader, eval_loader, train_dataset, eval_dataset, stats = build_clean_dataloaders(
        raw_dataset, batch_size=batch_size, T=T, N=N, F=F
    )

    logger.info("Creating diffusion model on device=%s", device)
    model = create_model(device, T=T, N=N, F=F, cond_dim=4)

    logger.info("Training diffusion model for %d epochs", num_epochs)
    best_ckpt, last_ckpt, history = train_diffusion_model(
        model,
        train_loader,
        device,
        num_epochs=num_epochs,
        save_dir=str(ROOT / "checkpoints"),
    )

    hist_df = pd.DataFrame(history)
    hist_path = OUTPUT_DIR / "training_history.csv"
    safe_save(hist_df, hist_path, force=force)

    logger.info("Loading best checkpoint for evaluation: %s", best_ckpt)
    eval_model = load_eval_model(best_ckpt, device, T=T, N=N, F=F, cond_dim=4)

    logger.info(
        "Sampling futures: num_samples=%d, num_steps=%d", num_samples, num_steps
    )
    samples = sample_future(
        eval_model,
        eval_loader,
        device,
        T=T,
        N=N,
        F=F,
        num_samples=num_samples,
        num_steps=num_steps,
    )

    if samples is None:
        raise RuntimeError("No samples produced by sample_future().")

    S, B_total, D = samples.shape
    logger.info("Samples shape: S=%d, B_total=%d, D=%d", S, B_total, D)

    # For a simple CSV, take first sample across S
    df_samples = pd.DataFrame(samples[0])
    simple_path = OUTPUT_DIR / "safety_eval_diffusion_simple.csv"
    safe_save(df_samples, simple_path, force=force)

    # Map back to meta rows corresponding to eval set
    eval_indices = (
        eval_dataset.indices
        if hasattr(eval_dataset, "indices")
        else eval_dataset.dataset.indices
    )
    meta_eval = meta_df.iloc[eval_indices].reset_index(drop=True)

    B_eval = df_samples.shape[0]
    if B_eval != len(meta_eval):
        logger.warning(
            "Mismatch between B_eval (%d) and meta_eval length (%d); truncating to min.",
            B_eval,
            len(meta_eval),
        )
    B_used = min(B_eval, len(meta_eval))

    traj_pred = (
        df_samples.to_numpy(dtype=np.float32)[:B_used].reshape(B_used, T, N, F)
    )
    logger.debug("traj_pred shape: %s", traj_pred.shape)

    # Build event-level records with exponential PET-based risk mapping
    records = []
    for i in range(B_used):
        meta_i = meta_eval.iloc[i]
        pet_model = float(meta_i.get("PET", float("nan")))

        risk_model = pet_to_risk_exponential(pet_model, risk_half_life)

        if np.isfinite(pet_model):
            if pet_model < 0.5:
                severity = "critical"
            elif pet_model < 1.0:
                severity = "high"
            else:
                severity = "medium"
        else:
            severity = None

        records.append(
            {
                "idx": int(i),
                "obj_i": meta_i.get("obj_i", None),
                "obj_j": meta_i.get("obj_j", None),
                "cell_id": meta_i.get("cell_id", None),
                "PET": pet_model,
                "risk_score": risk_model,
                "severity_evt": severity,
            }
        )

    df_events_model = pd.DataFrame(records)
    events_path = OUTPUT_DIR / "safety_events_diffusion_model.csv"
    safe_save(df_events_model, events_path, force=force)

    summary_model = compute_safety_metrics(df_events_model)
    df_summary_model = pd.DataFrame([summary_model])
    summary_path = OUTPUT_DIR / "safety_eval_diffusion_summary.csv"
    safe_save(df_summary_model, summary_path, force=force)

    return hist_df, df_events_model, df_summary_model


def main() -> None:
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    run_safety_eval_pipeline(
        T=args.T,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        risk_half_life=args.risk_half_life,
        force=args.force,
    )


if __name__ == "__main__":
    main()
