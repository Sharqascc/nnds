import os
import sys

import cv2
import matplotlib.pyplot as plt


def show_frame(video_path, title="Video Frame", frame_idx=0):
    """
    Display a specific frame from a video file.

    Parameters
    ----------
    video_path : str or os.PathLike
        Path to the video file. Must exist and be readable.
    title : str, optional
        Title for the displayed frame.
    frame_idx : int, optional
        Index of the frame to display (0‑based). Must be a non‑negative integer.

    Raises
    ------
    TypeError
        If ``video_path`` is not a string or path‑like object.
    FileNotFoundError
        If the file does not exist.
    ValueError
        If ``frame_idx`` is not a non‑negative integer.
    IOError
        If OpenCV cannot open the video file.
    RuntimeError
        If the requested frame cannot be read.
    """
    # ---- Validate video_path -------------------------------------------------
    if not isinstance(video_path, (str, bytes, os.PathLike)):
        raise TypeError("video_path must be a string or path‑like object")
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # ---- Validate frame_idx --------------------------------------------------
    if not isinstance(frame_idx, int) or frame_idx < 0:
        raise ValueError("frame_idx must be a non‑negative integer")

    # ---- Open video ---------------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise OSError(f"Unable to open video file: {video_path}")

    # ---- Seek to the requested frame ----------------------------------------
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Could not read frame {frame_idx} from video.")

    # ---- Display -------------------------------------------------------------
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()
    plt.pause(0.1)


def show_image(image, title="Image"):
    """Display an image (BGR format) using matplotlib."""
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()
    plt.pause(0.1)


def ask_user(prompt="Continue? (y/n): "):
    """Prompt the user for a yes/no answer."""
    print("\n" + "=" * 50)
    print("⏳ WAITING FOR YOUR INPUT")
    print(prompt)
    sys.stdout.flush()
    while True:
        ans = input().strip().lower()
        if ans in ("y", "yes", "n", "no"):
            return ans in ("y", "yes")
        print("Please answer 'y' or 'n'.")
