
import numpy as np
import cv2

H = np.load("/content/nnds_verify/calibration/H_from_giti_calibration_points.npy")

def pixel_to_world(px, py, H):
    pt = np.array([[[px, py]]], dtype=np.float32)
    w = cv2.perspectiveTransform(pt, H)
    return float(w[0,0,0]), float(w[0,0,1])

track = [
    {"frame": 0, "x": 1151, "y": 413},
    {"frame": 1, "x": 1149, "y": 414},
    {"frame": 2, "x": 1147, "y": 416},
    {"frame": 3, "x": 1145, "y": 417},
    {"frame": 4, "x": 1143, "y": 419},
    {"frame": 5, "x": 1141, "y": 420},
    {"frame": 6, "x": 1139, "y": 422},
    {"frame": 7, "x": 1137, "y": 423},
    {"frame": 8, "x": 1135, "y": 425},
    {"frame": 9, "x": 1133, "y": 426},
]
fps = 30.0
ref_speed_kmh = 24.0

p_start = track[0]
p_end = track[-1]

w_start = pixel_to_world(p_start["x"], p_start["y"], H)
w_end = pixel_to_world(p_end["x"], p_end["y"], H)

dist_raw = np.hypot(w_end[0]-w_start[0], w_end[1]-w_start[1])  # in whatever units H outputs
n_frames_elapsed = p_end["frame"] - p_start["frame"]
dt = n_frames_elapsed / fps

computed_speed_raw_units_per_s = dist_raw / dt

print(f"Raw world-space distance traveled: {dist_raw:.6f} (calibration units)")
print(f"Time elapsed: {dt:.4f} s")
print(f"Computed speed (assuming units = meters): {computed_speed_raw_units_per_s * 3.6:.3f} km/h")
print(f"Computed speed (assuming units = feet):   {computed_speed_raw_units_per_s * 0.3048 * 3.6:.3f} km/h")
print(f"Reference (ground truth) speed: {ref_speed_kmh} km/h")

scale_if_meters = ref_speed_kmh / (computed_speed_raw_units_per_s * 3.6)
scale_if_feet = ref_speed_kmh / (computed_speed_raw_units_per_s * 0.3048 * 3.6)
print(f"\nImplied correction factor if treating H output as meters: {scale_if_meters:.3f}x")
print(f"Implied correction factor if treating H output as feet:    {scale_if_feet:.3f}x")
