import json
import math
import os

import cv2
import numpy as np

VIDEO_PATH = "/content/nnds/sample_data/traffic_video.mp4"
CALIB_JSON = "/content/drive/MyDrive/camera_calibration/camera_intrinsics.json"
OUT_DIR = "/content/drive/MyDrive/camera_calibration/review_artifacts"
CONTACT_SHEET_PATH = os.path.join(OUT_DIR, "undistort_contact_sheet.jpg")
SAMPLE_VIDEO_PATH = os.path.join(OUT_DIR, "traffic_video_undistorted_sample.mp4")


def load_calibration(path):
    with open(path, "r") as f:
        data = json.load(f)
    K = np.array(data["video_space_calibration"]["matrix"], dtype=np.float64)
    dist = np.array(data["video_space_calibration"]["distortion_coefficients"], dtype=np.float64)
    return data, K, dist


def get_video_info(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return width, height, frame_count, fps


def sample_indices(frame_count, n=6):
    if frame_count <= 0:
        return [0]
    if frame_count <= n:
        return list(range(frame_count))
    return sorted(set(np.linspace(0, frame_count - 1, n, dtype=int).tolist()))


def draw_label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 44), (0, 0, 0), -1)
    cv2.putText(out, text, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def build_contact_sheet(images, cols=2, pad=12, bg=(18, 18, 18)):
    h, w = images[0].shape[:2]
    rows = math.ceil(len(images) / cols)
    sheet_h = rows * h + (rows + 1) * pad
    sheet_w = cols * w + (cols + 1) * pad
    sheet = np.full((sheet_h, sheet_w, 3), bg, dtype=np.uint8)

    for i, img in enumerate(images):
        r = i // cols
        c = i % cols
        y = pad + r * (h + pad)
        x = pad + c * (w + pad)
        sheet[y:y+h, x:x+w] = img

    return sheet


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    data, K, dist = load_calibration(CALIB_JSON)
    width, height, frame_count, fps = get_video_info(VIDEO_PATH)

    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (width, height), 0, (width, height))

    cap = cv2.VideoCapture(VIDEO_PATH)
    indices = sample_indices(frame_count, n=6)
    sheet_images = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue

        undistorted = cv2.undistort(frame, K, dist, None, new_K)

        x, y, w, h = roi
        if w > 0 and h > 0:
            undistorted = undistorted[y:y+h, x:x+w]
            undistorted = cv2.resize(undistorted, (width, height))

        original_labeled = draw_label(frame, f"Original frame {idx}")
        undistorted_labeled = draw_label(undistorted, f"Undistorted frame {idx}")
        pair = np.hstack([original_labeled, undistorted_labeled])
        sheet_images.append(pair)

    cap.release()

    if not sheet_images:
        raise RuntimeError("No frames were extracted for contact sheet.")

    contact_sheet = build_contact_sheet(sheet_images, cols=1, pad=16)
    cv2.imwrite(CONTACT_SHEET_PATH, contact_sheet)

    cap = cv2.VideoCapture(VIDEO_PATH)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(SAMPLE_VIDEO_PATH, fourcc, fps if fps > 0 else 30.0, (width, height))

    max_frames = min(frame_count, int((fps if fps > 0 else 30) * 10))
    written = 0

    for _ in range(max_frames):
        ok, frame = cap.read()
        if not ok:
            break
        undistorted = cv2.undistort(frame, K, dist, None, new_K)
        x, y, w, h = roi
        if w > 0 and h > 0:
            undistorted = undistorted[y:y+h, x:x+w]
            undistorted = cv2.resize(undistorted, (width, height))
        out.write(undistorted)
        written += 1

    cap.release()
    out.release()

    print(json.dumps({
        "video_resolution": [width, height],
        "frame_count": frame_count,
        "fps": fps,
        "roi": list(map(int, roi)),
        "frames_used_for_contact_sheet": indices,
        "contact_sheet_path": CONTACT_SHEET_PATH,
        "sample_video_path": SAMPLE_VIDEO_PATH,
        "sample_video_frames_written": written,
        "aspect_ratio_warning": data["aspect_ratio_check"]["same_aspect_ratio"] is False
    }, indent=2))


if __name__ == "__main__":
    main()
