
import os
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np


def _maybe_save(out_path: Optional[str]):
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")


def plot_pet_like_histogram(pet_pairs: List[Tuple[float, float]],
                            out_path: Optional[str] = None,
                            title: str = "PET-like step differences (sample - real)"):
    """
    Plot a histogram of PET-like step differences between sampled and real trajectories.

    Parameters
    ----------
    pet_pairs : list of (pet_real_steps, pet_sample_steps)
        Output from compute_pet_like_metrics.
    out_path : str or None
        If provided, save the figure to this path.
    title : str
        Title for the plot.
    """
    diffs = [
        (ps - pr)
        for (pr, ps) in pet_pairs
        if pr is not None and ps is not None
    ]
    if not diffs:
        print("No examples with both real and sample PET-like defined.")
        return

    diffs = np.array(diffs, dtype=float)

    plt.figure(figsize=(6, 4))
    plt.hist(diffs, bins=20, edgecolor="black", alpha=0.7)
    plt.axvline(0.0, color="red", linestyle="--", label="zero diff")
    plt.xlabel("sample PET-like steps - real PET-like steps")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    _maybe_save(out_path)
    plt.show()


def plot_true_vs_pet_like(records,
                          out_path: Optional[str] = None,
                          title: str = "True PET (s) vs PET-like steps"):
    """
    Scatter plot comparing true PET (seconds) from CSV and PET-like steps
    for real vs sampled trajectories.

    Parameters
    ----------
    records : list of (row_idx, true_pet, pet_like_real, pet_like_sample)
        Output from compare_realPET_samplePET.
    out_path : str or None
        If provided, save the figure to this path.
    title : str
        Title for the plot.
    """
    true_pet = []
    pet_real = []
    pet_sample = []

    for row_idx, t_pet, pr, ps in records:
        if pr is None or ps is None:
            continue
        true_pet.append(float(t_pet))
        pet_real.append(float(pr))
        pet_sample.append(float(ps))

    if not true_pet:
        print("No records with both real and sample PET-like defined.")
        return

    true_pet = np.array(true_pet)
    pet_real = np.array(pet_real)
    pet_sample = np.array(pet_sample)

    plt.figure(figsize=(6, 5))
    plt.scatter(true_pet, pet_real, s=20, alpha=0.7, label="PET-like real (steps)")
    plt.scatter(true_pet, pet_sample, s=20, alpha=0.7, label="PET-like sample (steps)")

    plt.xlabel("True PET from CSV (seconds)")
    plt.ylabel("PET-like (steps)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    _maybe_save(out_path)
    plt.show()


def plot_true_vs_sample_delta(records,
                              out_path: Optional[str] = None,
                              title: str = "True PET (s) vs (sample-real) PET-like steps"):
    """
    Plot true PET (seconds) vs difference in PET-like steps (sample - real).

    Parameters
    ----------
    records : list of (row_idx, true_pet, pet_like_real, pet_like_sample)
    out_path : str or None
    title : str
    """
    true_pet = []
    delta_steps = []

    for row_idx, t_pet, pr, ps in records:
        if pr is None or ps is None:
            continue
        true_pet.append(float(t_pet))
        delta_steps.append(float(ps - pr))

    if not true_pet:
        print("No records with both real and sample PET-like defined.")
        return

    true_pet = np.array(true_pet)
    delta_steps = np.array(delta_steps)

    plt.figure(figsize=(6, 4))
    plt.scatter(true_pet, delta_steps, s=20, alpha=0.7)
    plt.axhline(0.0, color="red", linestyle="--", label="zero diff")
    plt.xlabel("True PET from CSV (seconds)")
    plt.ylabel("sample-real PET-like (steps)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    _maybe_save(out_path)
    plt.show()
