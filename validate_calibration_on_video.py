import json
import os

import cv2
import numpy as np

VIDEO_PATH = "/content/nnds/sample_data/traffic_video.mp4"
CALIB_JSON = "/content/drive/MyDrive/camera_calibration/camera_intrinsics.json"
OUT_DIR = "/content/drive/MyDrive/camera_calibration/validation_frames"


def load_calibration(path):
    with open(path, "r") as f:
        data = json.load(f)

    K = np.array(data["video_space_calibration"]["matrix"], dtype=np.float64)
    dist = np.array(data["video_space_calibration"]["distortion_coefficients"], dtype=np.float64)
    return data, K, dist


def sample_frame_indices(frame_count, n=3):
    if frame_count <= 0:
        return [0]
    if frame_count < n:
        return list(range(frame_count))
    return sorted(set([0, frame_count // 2, max(0, frame_count - 1)]))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    data, K, dist = load_calibration(CALIB_JSON)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (width, height), 0, (width, height))

    indices = sample_frame_indices(frame_count, n=3)
    saved = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue

        undistorted = cv2.undistort(frame, K, dist, None, new_K)

        x, y, w, h = roi
        if w > 0 and h > 0:
            undistorted_crop = undistorted[y:y+h, x:x+w]
            undistorted_crop = cv2.resize(undistorted_crop, (width, height))
        else:
            undistorted_crop = undistorted

        panel = np.hstack([frame, undistorted_crop])

        label_h = 50
        canvas = np.zeros((height + label_h, width * 2, 3), dtype=np.uint8)
        canvas[label_h:, :, :] = panel
        cv2.putText(canvas, "Original", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Undistorted", (width + 20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)

        out_path = os.path.join(OUT_DIR, f"frame_{idx:06d}_compare.jpg")
        cv2.imwrite(out_path, canvas)
        saved.append(out_path)

    cap.release()

    print(json.dumps({
        "video_resolution": [width, height],
        "frame_count": frame_count,
        "roi": list(map(int, roi)),
        "saved_files": saved,
        "aspect_ratio_warning": data["aspect_ratio_check"]["same_aspect_ratio"] is False
    }, indent=2))


if __name__ == "__main__":
    main()
