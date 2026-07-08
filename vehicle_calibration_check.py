
import numpy as np
import cv2

H = np.load("/content/nnds_verify/calibration/H_from_giti_calibration_points.npy")

def pixel_to_world(px, py, H):
    pt = np.array([[[px, py]]], dtype=np.float32)
    w = cv2.perspectiveTransform(pt, H)
    return float(w[0,0,0]), float(w[0,0,1])

observations = [
    {
        "label": "white sedan, near-center intersection (diagonal)",
        "p1_px": (478, 248),
        "p2_px": (592, 207),
        "known_dist_m": 4.62,  # sqrt(4.3^2 + 1.7^2), typical sedan L/W
    },
    {
        "label": "white hatchback, mid-frame (diagonal)",
        "p1_px": (893, 308),
        "p2_px": (958, 262),
        "known_dist_m": 4.14,  # sqrt(3.8^2 + 1.65^2), typical hatchback L/W
    },
    {
        "label": "white SUV, upper-right intersection (diagonal)",
        "p1_px": (1092, 250),
        "p2_px": (1183, 207),
        "known_dist_m": 5.03,  # sqrt(4.7^2 + 1.8^2), typical SUV L/W
    },
]

print(f"{'Label':<45} {'H-output dist':<15} {'Known (m)':<12} {'Scale needed':<12}")
scales = []
for obs in observations:
    w1 = pixel_to_world(*obs["p1_px"], H)
    w2 = pixel_to_world(*obs["p2_px"], H)
    d_raw = np.hypot(w1[0]-w2[0], w1[1]-w2[1])
    scale_needed = obs["known_dist_m"] / d_raw if d_raw > 0 else float("nan")
    scales.append(scale_needed)
    print(f"{obs['label']:<45} {d_raw:<15.5f} {obs['known_dist_m']:<12.2f} {scale_needed:<12.2f}")

scales = np.array(scales)
print(f"\\nMean scale factor: {scales.mean():.2f}x")
print(f"Std dev: {scales.std():.2f}")
print(f"Coefficient of variation: {scales.std()/scales.mean()*100:.1f}%")
print("\\nIf CV is low (<15-20%), a single global scale correction is likely sufficient.")
print("If CV is high, the homography is spatially inconsistent and needs a full re-fit with new, better-spread calibration points.")
