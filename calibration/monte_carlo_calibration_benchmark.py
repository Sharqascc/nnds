
import numpy as np
import cv2

IMG_W, IMG_H = 1920, 1080

fx, fy = 1600.0, 1600.0
cx, cy = IMG_W / 2.0, IMG_H / 2.0
K = np.array([[fx, 0.0, cx],
              [0.0, fy, cy],
              [0.0, 0.0, 1.0]], dtype=np.float32)

dist_coeffs = np.array([-0.12, 0.02, 0.0, 0.0, 0.0], dtype=np.float32)

NX, NY = 27, 12
W_X, W_Y = 49.5, 22.0
x_coords = np.linspace(0.0, W_X, NX)
y_coords = np.linspace(0.0, W_Y, NY)
XX, YY = np.meshgrid(x_coords, y_coords)
ZW = np.zeros_like(XX)
world_points_true = np.stack([XX.ravel(), YY.ravel(), ZW.ravel()], axis=1).astype(np.float32)

sigma_px_x = 2.0
sigma_px_y = 3.0
rho_noise = 0.83
plane_bias_cm = 1.5

def make_example_pose():
    rvec = np.array([0.4, 0.0, 0.0], dtype=np.float32)
    R, _ = cv2.Rodrigues(rvec)
    t = np.array([[0.0], [-15.0], [25.0]], dtype=np.float32)
    return R, t

def project_points(world_pts, R, t, K, dist):
    rvec, _ = cv2.Rodrigues(R)
    img_pts, _ = cv2.projectPoints(world_pts, rvec, t, K, dist)
    return img_pts.reshape(-1, 2)

R_true, t_true = make_example_pose()

def add_plane_bias(world_pts, bias_cm=1.5):
    bias_m = bias_cm / 100.0
    wp = world_pts.copy()
    x_range = np.ptp(wp[:, 0])
    x_norm = (wp[:, 0] - wp[:, 0].min()) / max(1e-6, x_range)
    wp[:, 2] = x_norm * bias_m
    return wp

def add_anisotropic_noise_2d(points_2d, rng, sx, sy, rho):
    N = points_2d.shape[0]
    cov = np.array([[sx**2, rho * sx * sy],
                    [rho * sx * sy, sy**2]], dtype=np.float32)
    noise = rng.multivariate_normal(mean=[0.0, 0.0], cov=cov, size=N)
    return points_2d + noise.astype(np.float32)

def mae_world(pred, gt):
    return float(np.mean(np.linalg.norm(pred - gt, axis=1)))

def world_from_pnp(R, t, K, dist, img_pts):
    img_pts_undist = cv2.undistortPoints(img_pts.reshape(-1, 1, 2), K, dist)
    img_pts_undist = img_pts_undist.reshape(-1, 2)
    rays_cam = np.concatenate(
        [img_pts_undist, np.ones((img_pts_undist.shape[0], 1), dtype=np.float32)],
        axis=1
    )
    R_inv = R.T
    t = t.reshape(3)
    cam_center_world = -R_inv @ t
    pts_world = []
    for d_cam in rays_cam:
        d_world = R_inv @ d_cam
        if abs(d_world[2]) < 1e-6:
            pts_world.append([np.nan, np.nan])
            continue
        s = -cam_center_world[2] / d_world[2]
        P_world = cam_center_world + s * d_world
        pts_world.append(P_world[:2])
    return np.array(pts_world, dtype=np.float32)

def estimate_homography(world_pts, img_pts):
    src = img_pts.astype(np.float32)
    dst = world_pts[:, :2].astype(np.float32)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    return H, mask

def apply_homography(H, img_pts):
    pts = img_pts.astype(np.float32)
    pts_h = cv2.convertPointsToHomogeneous(pts).reshape(-1, 3).T
    mapped = H @ pts_h
    mapped = (mapped[:2] / mapped[2]).T
    return mapped

def solve_pnp_world_error(world_pts, img_pts, method_flag):
    obj = world_pts.reshape(-1, 1, 3).astype(np.float32)
    img = img_pts.reshape(-1, 1, 2).astype(np.float32)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist_coeffs, flags=method_flag)
    if not ok:
        return np.inf, None, None
    R_est, _ = cv2.Rodrigues(rvec)
    world_est = world_from_pnp(R_est, tvec, K, dist_coeffs, img_pts)
    mae = mae_world(world_est, world_points_true[:, :2])
    return mae, R_est, tvec

def solve_pnp_Z0_world_error(world_pts, img_pts, method_flag):
    wp = world_pts.copy()
    wp[:, 2] = 0.0
    return solve_pnp_world_error(wp, img_pts, method_flag)

def solve_p3p_ransac_world_error(world_pts, img_pts, iterations=200, sample_size=4):
    if not hasattr(cv2, "SOLVEPNP_P3P"):
        return np.inf
    obj_all = world_pts.astype(np.float32)
    img_all = img_pts.astype(np.float32)
    N = obj_all.shape[0]
    best_mae = np.inf
    for _ in range(iterations):
        idx = np.random.choice(N, size=sample_size, replace=False)
        obj_sample = obj_all[idx].reshape(-1, 1, 3)
        img_sample = img_all[idx].reshape(-1, 1, 2)
        ok, rvec, tvec = cv2.solvePnP(
            obj_sample, img_sample, K, dist_coeffs, flags=cv2.SOLVEPNP_P3P
        )
        if not ok:
            continue
        R_est, _ = cv2.Rodrigues(rvec)
        world_est = world_from_pnp(R_est, tvec, K, dist_coeffs, img_all)
        mae = mae_world(world_est, world_points_true[:, :2])
        if mae < best_mae:
            best_mae = mae
    return best_mae

def run_single_trial(seed=0):
    rng = np.random.default_rng(seed)
    world_points_biased = add_plane_bias(world_points_true, plane_bias_cm)
    img_points_ideal = project_points(world_points_biased, R_true, t_true, K, dist_coeffs)
    img_points_noisy = add_anisotropic_noise_2d(
        img_points_ideal, rng, sigma_px_x, sigma_px_y, rho_noise
    )

    H_biased, _ = estimate_homography(world_points_biased, img_points_noisy)
    pred_world_h_biased = apply_homography(H_biased, img_points_noisy)
    mae_h_true_biased = mae_world(pred_world_h_biased, world_points_true[:, :2])

    H_Z0, _ = estimate_homography(world_points_true, img_points_noisy)
    pred_world_h_Z0 = apply_homography(H_Z0, img_points_noisy)
    mae_h_true_Z0 = mae_world(pred_world_h_Z0, world_points_true[:, :2])

    mae_pnp_iter, _, _ = solve_pnp_world_error(
        world_points_biased, img_points_noisy, cv2.SOLVEPNP_ITERATIVE
    )

    mae_pnp_iter_Z0, _, _ = solve_pnp_Z0_world_error(
        world_points_biased, img_points_noisy, cv2.SOLVEPNP_ITERATIVE
    )

    mae_p3p = solve_p3p_ransac_world_error(world_points_biased, img_points_noisy)
    return mae_h_true_biased, mae_h_true_Z0, mae_pnp_iter, mae_pnp_iter_Z0, mae_p3p

def run_monte_carlo(num_trials=50):
    mae_H_biased, mae_H_Z0, mae_PNP, mae_PNP_Z0, mae_P3P = [], [], [], [], []
    for s in range(num_trials):
        h_b, h_z0, p, pz0, p3p = run_single_trial(seed=s)
        mae_H_biased.append(h_b)
        mae_H_Z0.append(h_z0)
        mae_PNP.append(p)
        mae_PNP_Z0.append(pz0)
        mae_P3P.append(p3p)
    return (
        np.array(mae_H_biased),
        np.array(mae_H_Z0),
        np.array(mae_PNP),
        np.array(mae_PNP_Z0),
        np.array(mae_P3P),
    )

def summarize(name, arr):
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr))
    print(f"{name}: mean = {mean:.4f} m, std = {std:.4f} m")

if __name__ == "__main__":
    NUM_TRIALS = 50
    print(f"=== Monte Carlo results over {NUM_TRIALS} trials ===")
    mae_H_biased, mae_H_Z0, mae_PNP, mae_PNP_Z0, mae_P3P = run_monte_carlo(NUM_TRIALS)
    summarize("Homography (biased world)", mae_H_biased)
    summarize("Homography (Z=0 world)", mae_H_Z0)
    summarize("PnP ITERATIVE (biased Z)", mae_PNP)
    summarize("PnP ITERATIVE (Z=0)", mae_PNP_Z0)
    if np.isfinite(mae_P3P).any():
        summarize("P3P (RANSAC wrapper)", mae_P3P)
    else:
        print("P3P (RANSAC wrapper): not available / all inf")
