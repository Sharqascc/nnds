
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
ref_speed_ms = ref_speed_kmh / 3.6

# Check per-consecutive-frame speed AND per-pair segment speed
print("Per-frame-step raw distances and implied instantaneous speed (m/s), assuming H outputs meters:")
for i in range(len(track)-1):
    a, b = track[i], track[i+1]
    wa = pixel_to_world(a["x"], a["y"], H)
    wb = pixel_to_world(b["x"], b["y"], H)
    d = np.hypot(wb[0]-wa[0], wb[1]-wa[1])
    dt = (b["frame"] - a["frame"]) / fps
    speed = d / dt
    scale_needed = ref_speed_ms / speed if speed > 0 else float('nan')
    print(f"  frame {a['frame']}->{b['frame']}: dist={d:.6f}, speed={speed:.4f} m/s, scale_needed={scale_needed:.2f}x")

# Also check the calibration point spacing itself for internal consistency
print()
print("Pairwise world distances between calibration points (in native units, likely feet):")
cal_points = [
    {"id":1, "px": (947,715), "world": (222006.49, 730911.81)},
    {"id":2, "px": (194,217), "world": (222005.64, 730910.20)},
    {"id":3, "px": (1455,274), "world": (222007.06, 730910.70)},
    {"id":4, "px": (1136,155), "world": (222006.68, 730910.26)},
    {"id":5, "px": (714,221), "world": (222006.30, 730910.37)},
    {"id":6, "px": (997,124), "world": (222006.49, 730910.12)},
]
import itertools
for a, b in itertools.combinations(cal_points, 2):
    dn = a["world"][0]-b["world"][0]
    de = a["world"][1]-b["world"][1]
    d_world = np.hypot(dn, de)
    dpx = np.hypot(a["px"][0]-b["px"][0], a["px"][1]-b["px"][1])
    print(f"  pt{a['id']}-pt{b['id']}: pixel_dist={dpx:.1f}px, world_dist={d_world:.3f} (native units), px_per_native_unit={dpx/d_world if d_world>0 else float('nan'):.2f}")
