import json
import numpy as np
import cv2

def load_giti_homography(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    pts_pix = []
    pts_world = []

    for p in data["calibration_points"]:
        px = p["pixel"]["x"]
        py = p["pixel"]["y"]
        X = p["world"]["easting"]
        Y = p["world"]["northing"]
        pts_pix.append([px, py])
        pts_world.append([X, Y])

    pts_pix = np.array(pts_pix, dtype=np.float32)
    pts_world = np.array(pts_world, dtype=np.float32)

    H, mask = cv2.findHomography(pts_pix, pts_world, cv2.RANSAC, ransacReprojThreshold=5.0)
    return H, mask, pts_world
