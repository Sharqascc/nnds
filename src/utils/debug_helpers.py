"""
Beginner-friendly debugging helpers for NNDS.

Usage in any script:
    from src.utils.debug_helpers import debug_save_image, debug_save_df, debug_print_section

    debug_print_section("Detections summary")
    debug_save_image(frame, "frame_001.png", subdir="debug")
    debug_save_df(df, "detections.csv", subdir="debug")
"""

import pathlib
import traceback
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEBUG_ROOT = pathlib.Path("str(Path(__file__).resolve().parents[1])/outputs/debug")


def _ensure_dir(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def debug_print_section(title: str, details: dict[str, Any] | None = None) -> None:
    """Print a clearly delimited debug section."""
    print("\n" + "=" * 60)
    print(f"🔍 DEBUG: {title}")
    if details:
        for k, v in details.items():
            print(f"  {k}: {v}")
    print("=" * 60)


def debug_save_image(
    image: np.ndarray,
    filename: str,
    subdir: str = "debug",
    title: str = "",
    show: bool = False,
) -> pathlib.Path:
    """
    Save an image to outputs/debug/<subdir>/<filename>.
    Optionally display it in Colab.

    image: HxW or HxWx3 (BGR or RGB). If BGR from OpenCV, pass bgr=True if you want auto-convert.
    """
    out_dir = DEBUG_ROOT / subdir
    _ensure_dir(out_dir)
    out_path = out_dir / filename

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[-1] == 3:
        # Assume BGR from OpenCV; convert to RGB for matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image

    cv2.imwrite(str(out_path), image)

    if show:
        plt.figure(figsize=(8, 6))
        plt.imshow(image_rgb if image_rgb is not locals().get("image_rgb") else image)
        if title:
            plt.title(title)
        plt.axis("off")
        plt.show()

    return out_path


def debug_save_df(
    df: pd.DataFrame,
    filename: str,
    subdir: str = "debug",
    head_lines: int = 10,
) -> pathlib.Path:
    """Save a DataFrame to CSV and print a small preview."""
    out_dir = DEBUG_ROOT / subdir
    _ensure_dir(out_dir)
    out_path = out_dir / filename

    df.to_csv(out_path, index=False)

    debug_print_section(f"Saved DataFrame to {out_path}")
    print("First rows:")
    print(df.head(head_lines).to_string())
    print("\nShape:", df.shape)
    print("Columns:", list(df.columns))

    return out_path


def debug_print_tensor_info(name: str, tensor: Any) -> None:
    """Print basic info about a torch-like tensor (shape, dtype, NaN/Inf)."""
    try:
        import torch

        is_tensor = torch.is_tensor(tensor)
    except Exception:
        is_tensor = hasattr(tensor, "shape") and hasattr(tensor, "dtype")

    if not is_tensor:
        debug_print_section(f"{name}: not a tensor")
        return

    arr = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)

    debug_print_section(
        f"{name} info",
        {
            "shape": arr.shape,
            "dtype": arr.dtype,
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
            "has_nan": bool(np.isnan(arr).any()),
            "has_inf": bool(np.isinf(arr).any()),
        },
    )


def debug_traceback(e: Exception) -> None:
    """Print a friendly traceback summary for beginners."""
    debug_print_section("Error occurred")
    print("Exception type:", type(e).__name__)
    print("Message:", str(e))
    print("\nTraceback (last 10 frames):")
    tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
    print("".join(tb_lines[-10:]))
