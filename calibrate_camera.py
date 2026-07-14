import glob
import json
import os
from pathlib import Path

import cv2
import numpy as np

BOARD_COLS = 9
BOARD_ROWS = 6
SQUARE_SIZE_MM = 25.0

PHOTOS_DIR = "/content/drive/MyDrive/camera_calibration"
VIDEO_PATH = "/content/nnds/sample_data/traffic_video.mp4"
OUTPUT_JSON = "/content/drive/MyDrive/camera_calibration/camera_intrinsics.json"


def get_video_resolution(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video resolution from {video_path}: {width}x{height}")
    return width, height


def build_object_points():
    objp = np.zeros((BOARD_ROWS * BOARD_COLS, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS].T.reshape(-1, 2).astype(np.float32)
    objp *= np.float32(SQUARE_SIZE_MM)
    return objp.reshape(-1, 1, 3)


def collect_image_paths():
    paths = sorted(
        glob.glob(os.path.join(PHOTOS_DIR, "*.jpg")) +
        glob.glob(os.path.join(PHOTOS_DIR, "*.jpeg")) +
        glob.glob(os.path.join(PHOTOS_DIR, "*.png"))
    )
    if not paths:
        raise FileNotFoundError(f"No checkerboard photos found in {PHOTOS_DIR}")
    return paths


def detect_corners():
    template_objp = build_object_points()
    objpoints, imgpoints = [], []
    image_paths = collect_image_paths()

    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    img_shape = None
    accepted = []
    rejected = []

    print(f"Found {len(image_paths)} images.")

    for path in image_paths:
        img = cv2.imread(path)
        name = os.path.basename(path)

        if img is None:
            print(f"[skip] unreadable {name}")
            rejected.append({"file": name, "reason": "unreadable"})
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(gray, (BOARD_COLS, BOARD_ROWS), flags)

        if not found:
            print(f"[skip] no checkerboard in {name}")
            rejected.append({"file": name, "reason": "not_detected"})
            continue

        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(np.ascontiguousarray(template_objp.copy(), dtype=np.float32))
        imgpoints.append(np.ascontiguousarray(corners.reshape(-1, 1, 2), dtype=np.float32))
        accepted.append(name)
        print(f"[ok]   {name}")

    print(f"\nUsed {len(objpoints)}/{len(image_paths)} images.")

    if len(objpoints) == 0:
        raise RuntimeError("No valid checkerboard detections found.")

    return objpoints, imgpoints, img_shape, accepted, rejected


def calibrate_camera(objpoints, imgpoints, img_shape):
    objpoints = [np.ascontiguousarray(x, dtype=np.float32) for x in objpoints]
    imgpoints = [np.ascontiguousarray(x, dtype=np.float32) for x in imgpoints]

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )

    total_error = 0.0
    per_view_errors = []

    for i in range(len(objpoints)):
        projected, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        err = cv2.norm(imgpoints[i], projected, cv2.NORM_L2) / len(projected)
        per_view_errors.append(float(err))
        total_error += err

    mean_error = total_error / len(objpoints)
    return camera_matrix, dist_coeffs, float(mean_error), per_view_errors


def scale_intrinsics(camera_matrix, from_shape, to_shape):
    from_w, from_h = from_shape
    to_w, to_h = to_shape

    sx = to_w / from_w
    sy = to_h / from_h

    scaled = camera_matrix.copy()
    scaled[0, 0] *= sx
    scaled[0, 2] *= sx
    scaled[1, 1] *= sy
    scaled[1, 2] *= sy
    return scaled, sx, sy


def aspect_ratio_report(calib_shape, video_shape):
    cw, ch = calib_shape
    vw, vh = video_shape
    calib_ar = cw / ch
    video_ar = vw / vh
    diff = abs(calib_ar - video_ar)
    same_aspect = diff < 1e-3
    return {
        "calibration_resolution": {"width": cw, "height": ch},
        "video_resolution": {"width": vw, "height": vh},
        "calibration_aspect_ratio": calib_ar,
        "video_aspect_ratio": video_ar,
        "absolute_difference": diff,
        "same_aspect_ratio": same_aspect,
    }


def matrix_to_dict(K):
    return {
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "matrix": K.tolist(),
    }


def main():
    video_w, video_h = get_video_resolution(VIDEO_PATH)
    objpoints, imgpoints, img_shape, accepted, rejected = detect_corners()
    camera_matrix, dist_coeffs, mean_error, per_view_errors = calibrate_camera(
        objpoints, imgpoints, img_shape
    )

    scaled_matrix, sx, sy = scale_intrinsics(camera_matrix, img_shape, (video_w, video_h))
    aspect = aspect_ratio_report(img_shape, (video_w, video_h))

    result = {
        "checkerboard": {
            "board_cols_internal_corners": BOARD_COLS,
            "board_rows_internal_corners": BOARD_ROWS,
            "square_size_mm": SQUARE_SIZE_MM,
        },
        "source_paths": {
            "photos_dir": PHOTOS_DIR,
            "video_path": VIDEO_PATH,
        },
        "image_detection_summary": {
            "num_images_total": len(accepted) + len(rejected),
            "num_images_used": len(accepted),
            "accepted_files": accepted,
            "rejected_files": rejected,
        },
        "quality": {
            "mean_reprojection_error_px": mean_error,
            "per_view_reprojection_error_px": per_view_errors,
        },
        "raw_calibration": {
            "resolution": {"width": img_shape[0], "height": img_shape[1]},
            **matrix_to_dict(camera_matrix),
            "distortion_coefficients": dist_coeffs.ravel().tolist(),
        },
        "video_space_calibration": {
            "resolution": {"width": video_w, "height": video_h},
            **matrix_to_dict(scaled_matrix),
            "distortion_coefficients": dist_coeffs.ravel().tolist(),
            "scaling": {"sx": sx, "sy": sy},
        },
        "aspect_ratio_check": aspect,
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(result, f, indent=2)

    print("\n--- Calibration summary ---")
    print(json.dumps({
        "video_resolution": [video_w, video_h],
        "calibration_resolution": list(img_shape),
        "num_images_used": len(accepted),
        "mean_reprojection_error_px": mean_error,
        "same_aspect_ratio": aspect["same_aspect_ratio"],
        "output_json": OUTPUT_JSON,
    }, indent=2))

    if not aspect["same_aspect_ratio"]:
        print("\nWARNING: calibration photos and video have different aspect ratios.")
        print("Scaled intrinsics are saved, but if the video is cropped or captured in a different camera mode,")
        print("the raw calibration may not transfer perfectly.")

    print(f"\nSaved intrinsics to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
