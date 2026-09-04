import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.bev.calibration.monte_carlo_calibration_benchmark import (
    add_anisotropic_noise_2d,
    add_plane_bias,
    apply_homography,
    compare_methods,
    configure_logging,
    estimate_homography,
    export_summary,
    mae_world,
    main,
    make_example_pose,
    maybe_plot_mae_hist,
    parse_args,
    project_points,
    run_monte_carlo,
    run_single_scenario,
    run_single_trial,
    solve_p3p_ransac_world_error,
    solve_pnp_world_error,
    solve_pnp_Z0_world_error,
    summarize,
    world_from_pnp,
)


# ---------------- parse_args ----------------
def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog"])
    parser = parse_args()
    args = parser.parse_args()
    assert args.num_trials == 50
    assert args.seed == 0
    assert args.plot is False
    assert args.multi_noise is False


def test_parse_args_custom(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--num-trials", "5", "--seed", "123", "--plot", "--multi-noise", "--verbose"],
    )
    parser = parse_args()
    args = parser.parse_args()
    assert args.num_trials == 5
    assert args.seed == 123
    assert args.plot is True
    assert args.multi_noise is True
    assert args.verbose is True


# ---------------- configure_logging ----------------
def test_configure_logging():
    configure_logging(verbose=True)
    configure_logging(verbose=False)


# ---------------- make_example_pose ----------------
def test_make_example_pose():
    R, t = make_example_pose()
    assert R.shape == (3, 3)
    assert t.shape == (3, 1)


# ---------------- project_points ----------------
def test_project_points():
    R, t = make_example_pose()
    pts = np.array([[0, 0, 0], [1, 1, 0]], dtype=np.float32)
    img_pts = project_points(pts, R, t, np.eye(3, dtype=np.float32), np.zeros(5, dtype=np.float32))
    assert img_pts.shape == (2, 2)


# ---------------- add_plane_bias ----------------
def test_add_plane_bias():
    pts = np.array([[0, 0, 0], [10, 0, 0]], dtype=np.float32)
    biased = add_plane_bias(pts, bias_cm=1.0)
    assert biased.shape == pts.shape
    assert biased[1, 2] > biased[0, 2]


# ---------------- add_anisotropic_noise_2d ----------------
def test_add_anisotropic_noise_2d():
    rng = np.random.default_rng(0)
    pts = np.array([[10, 20], [30, 40]], dtype=np.float32)
    noisy = add_anisotropic_noise_2d(pts, rng, sx=2.0, sy=3.0, rho=0.5)
    assert noisy.shape == pts.shape
    assert not np.allclose(noisy, pts)


# ---------------- mae_world ----------------
def test_mae_world():
    pred = np.array([[0, 0], [1, 1]], dtype=np.float32)
    gt = np.array([[0, 0], [2, 1]], dtype=np.float32)
    assert mae_world(pred, gt) == pytest.approx(0.5)


# ---------------- world_from_pnp ----------------
def test_world_from_pnp_identity():
    R = np.eye(3, dtype=np.float32)
    t = np.zeros((3, 1), dtype=np.float32)
    K_ = np.eye(3, dtype=np.float32)
    dist = np.zeros(5, dtype=np.float32)
    img_pts = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
    world = world_from_pnp(R, t, K_, dist, img_pts)
    assert world.shape == (3, 2)
    assert np.all(np.isfinite(world))


# ---------------- estimate_homography ----------------
def test_estimate_homography():
    world = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
    img = np.array([[100, 100], [200, 100], [100, 200], [200, 200]], dtype=np.float32)
    H = np.eye(3, dtype=np.float32)
    mask = np.ones((4, 1), dtype=np.uint8)
    with patch("cv2.findHomography", return_value=(H, mask)):
        H_out, mask_out = estimate_homography(world, img)
    assert H_out is H
    assert mask_out is mask


# ---------------- apply_homography ----------------
def test_apply_homography_identity():
    H = np.eye(3, dtype=np.float32)
    pts = np.array([[1, 2], [3, 4]], dtype=np.float32)
    mapped = apply_homography(H, pts)
    assert np.allclose(mapped, pts)


# ---------------- solve_pnp_world_error ----------------
def test_solve_pnp_world_error_failure():
    world = np.zeros((5, 3), dtype=np.float32)
    img = np.zeros((5, 2), dtype=np.float32)
    with patch("cv2.solvePnP", return_value=(False, None, None)):
        mae, R, t = solve_pnp_world_error(world, img, 0)
    assert mae == float("inf")
    assert R is None
    assert t is None


def test_solve_pnp_world_error_success():
    world = np.zeros((5, 3), dtype=np.float32)
    img = np.zeros((5, 2), dtype=np.float32)
    with (
        patch("cv2.solvePnP", return_value=(True, np.zeros(3), np.zeros(3))),
        patch("cv2.Rodrigues", return_value=(np.eye(3, dtype=np.float32), None)),
        patch(
            "src.bev.calibration.monte_carlo_calibration_benchmark.world_from_pnp",
            return_value=world[:, :2],
        ),
        patch(
            "src.bev.calibration.monte_carlo_calibration_benchmark.mae_world", return_value=0.123
        ),
    ):
        mae, R, t = solve_pnp_world_error(world, img, 0)
    assert mae == 0.123
    assert R is not None
    assert t is not None


# ---------------- solve_pnp_Z0_world_error ----------------
def test_solve_pnp_Z0_world_error():
    world = np.array([[0, 0, 1], [1, 1, 2]], dtype=np.float32)
    img = np.array([[0, 0], [1, 1]], dtype=np.float32)
    with patch(
        "src.bev.calibration.monte_carlo_calibration_benchmark.solve_pnp_world_error",
        return_value=(0.5, None, None),
    ) as mock_solve:
        mae, R, t = solve_pnp_Z0_world_error(world, img, 0)
    # Verify Z set to zero before solve
    called_wp = mock_solve.call_args[0][0]
    assert np.all(called_wp[:, 2] == 0.0)
    assert mae == 0.5


# ---------------- solve_p3p_ransac_world_error ----------------
def test_p3p_not_available(monkeypatch):
    import cv2

    monkeypatch.delattr(cv2, "SOLVEPNP_P3P", raising=False)
    assert solve_p3p_ransac_world_error(np.zeros((4, 3)), np.zeros((4, 2))) == float("inf")


def test_p3p_with_failures_and_success():
    world = np.zeros((5, 3), dtype=np.float32)
    img = np.zeros((5, 2), dtype=np.float32)
    rng = np.random.default_rng(0)
    with (
        patch("cv2.SOLVEPNP_P3P", 1, create=True),
        patch("cv2.solvePnP", side_effect=[(False, None, None), (True, np.zeros(3), np.zeros(3))]),
        patch("cv2.Rodrigues", return_value=(np.eye(3, dtype=np.float32), None)),
        patch(
            "src.bev.calibration.monte_carlo_calibration_benchmark.world_from_pnp",
            return_value=world[:, :2],
        ),
        patch("src.bev.calibration.monte_carlo_calibration_benchmark.mae_world", return_value=0.8),
    ):
        best = solve_p3p_ransac_world_error(world, img, iterations=2, sample_size=4, rng=rng)
    assert best == 0.8


# ---------------- run_single_trial ----------------
def test_run_single_trial(monkeypatch):
    # Patch all sub-functions used in run_single_trial to return deterministic values
    import src.bev.calibration.monte_carlo_calibration_benchmark as mc

    monkeypatch.setattr(
        mc, "add_plane_bias", MagicMock(return_value=np.zeros((27 * 12, 3), dtype=np.float32))
    )
    monkeypatch.setattr(
        mc, "project_points", MagicMock(return_value=np.zeros((27 * 12, 2), dtype=np.float32))
    )
    monkeypatch.setattr(
        mc,
        "add_anisotropic_noise_2d",
        MagicMock(return_value=np.zeros((27 * 12, 2), dtype=np.float32)),
    )
    monkeypatch.setattr(
        mc, "estimate_homography", MagicMock(side_effect=[(np.eye(3), None), (np.eye(3), None)])
    )
    monkeypatch.setattr(
        mc, "apply_homography", MagicMock(return_value=np.zeros((27 * 12, 2), dtype=np.float32))
    )
    monkeypatch.setattr(mc, "mae_world", MagicMock(return_value=0.1))
    monkeypatch.setattr(mc, "solve_pnp_world_error", MagicMock(return_value=(0.2, None, None)))
    monkeypatch.setattr(mc, "solve_pnp_Z0_world_error", MagicMock(return_value=(0.3, None, None)))
    monkeypatch.setattr(mc, "solve_p3p_ransac_world_error", MagicMock(return_value=0.4))

    res = run_single_trial(0, 1.0, 2.0, 0.5, 1.0)
    assert len(res) == 5
    assert res[0] == 0.1
    assert res[1] == 0.1
    assert res[2] == 0.2
    assert res[3] == 0.3
    assert res[4] == 0.4


# ---------------- run_monte_carlo ----------------
def test_run_monte_carlo(monkeypatch):
    import src.bev.calibration.monte_carlo_calibration_benchmark as mc

    # Patch tqdm to identity
    monkeypatch.setattr(mc, "tqdm", lambda x, **kwargs: x)

    def fake_single(seed, **kwargs):
        return (0.1, 0.2, 0.3, 0.4, 0.5)

    monkeypatch.setattr(mc, "run_single_trial", fake_single)
    arrays = run_monte_carlo(2, 0, 1.0, 2.0, 0.5, 1.0)
    assert all(len(a) == 2 for a in arrays)
    assert np.allclose(arrays[0], [0.1, 0.1])


# ---------------- summarize ----------------
def test_summarize():
    arr = np.array([1.0, 2.0, 3.0])
    mean, std = summarize("test", arr)
    assert mean == pytest.approx(2.0)
    assert std == pytest.approx(np.std(arr))


# ---------------- export_summary ----------------
def test_export_summary(tmp_path):
    arr = np.array([0.1, 0.2, 0.3])
    out = tmp_path / "summary.json"
    summary = export_summary(
        out,
        num_trials=3,
        sigma_px_x=1.0,
        sigma_px_y=2.0,
        rho_noise=0.5,
        plane_bias_cm=1.5,
        mae_H_biased=arr,
        mae_H_Z0=arr,
        mae_PNP=arr,
        mae_PNP_Z0=arr,
        mae_P3P=arr,
    )
    assert out.exists()
    assert summary["num_trials"] == 3
    assert "mae_stats" in summary


# ---------------- compare_methods ----------------
def test_compare_methods_inf():
    summary = {"mae_stats": {"H_biased_mean": 0.0, "PnP_iter_mean": 0.1}}
    comp = compare_methods(summary)
    assert comp["pnp_vs_homography_factor"] == float("inf")


def test_compare_methods_normal():
    summary = {"mae_stats": {"H_biased_mean": 0.2, "PnP_iter_mean": 0.1}}
    comp = compare_methods(summary)
    assert comp["pnp_vs_homography_factor"] == pytest.approx(0.5)


# ---------------- maybe_plot_mae_hist ----------------
def test_maybe_plot_mae_hist(tmp_path):
    arr = np.random.default_rng(0).normal(0, 1, 10)
    out = tmp_path / "hist.png"
    with (
        patch("matplotlib.pyplot.figure"),
        patch("matplotlib.pyplot.hist"),
        patch("matplotlib.pyplot.xlabel"),
        patch("matplotlib.pyplot.ylabel"),
        patch("matplotlib.pyplot.title"),
        patch("matplotlib.pyplot.grid"),
        patch("matplotlib.pyplot.tight_layout"),
        patch("matplotlib.pyplot.savefig"),
        patch("matplotlib.pyplot.close"),
    ):
        maybe_plot_mae_hist(arr, out)


# ---------------- run_single_scenario ----------------
def test_run_single_scenario_no_plot(tmp_path, monkeypatch):
    import src.bev.calibration.monte_carlo_calibration_benchmark as mc

    args = SimpleNamespace(
        num_trials=2, seed=0, output_summary=tmp_path / "summary.json", plot=False
    )
    # Patch run_monte_carlo to return arrays
    arr = np.array([0.1, 0.2])
    monkeypatch.setattr(mc, "run_monte_carlo", MagicMock(return_value=(arr, arr, arr, arr, arr)))
    monkeypatch.setattr(mc, "summarize", MagicMock(return_value=(0.1, 0.01)))
    # export_summary will be called, compare_methods too; let them run but maybe_plot skipped because plot False
    summary = run_single_scenario(args, 1.0, 2.0, 0.5, 1.0)
    assert "mae_stats" in summary
    assert "comparison" in summary


def test_run_single_scenario_with_plot(tmp_path, monkeypatch):
    import src.bev.calibration.monte_carlo_calibration_benchmark as mc

    args = SimpleNamespace(
        num_trials=2, seed=0, output_summary=tmp_path / "summary.json", plot=True
    )
    arr = np.array([0.1, 0.2])
    monkeypatch.setattr(mc, "run_monte_carlo", MagicMock(return_value=(arr, arr, arr, arr, arr)))
    monkeypatch.setattr(mc, "summarize", MagicMock(return_value=(0.1, 0.01)))
    monkeypatch.setattr(mc, "maybe_plot_mae_hist", MagicMock())
    summary = run_single_scenario(args, 1.0, 2.0, 0.5, 1.0, suffix=None)
    assert "comparison" in summary
    mc.maybe_plot_mae_hist.assert_called_once()


# ---------------- main ----------------
def test_main_single_scenario(monkeypatch, tmp_path):
    import src.bev.calibration.monte_carlo_calibration_benchmark as mc

    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--num-trials", "1", "--output-summary", str(tmp_path / "summary.json")],
    )
    monkeypatch.setattr(
        mc,
        "parse_args",
        lambda: SimpleNamespace(
            num_trials=1,
            seed=0,
            plot=False,
            multi_noise=False,
            output_summary=tmp_path / "summary.json",
            verbose=False,
        ),
    )
    monkeypatch.setattr(mc, "configure_logging", MagicMock())
    monkeypatch.setattr(mc, "run_single_scenario", MagicMock(return_value={"test": "ok"}))
    main()
    mc.run_single_scenario.assert_called_once()


def test_main_multi_noise(monkeypatch, tmp_path):
    import src.bev.calibration.monte_carlo_calibration_benchmark as mc

    output_summary = tmp_path / "summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--multi-noise", "--num-trials", "1", "--output-summary", str(output_summary)],
    )
    monkeypatch.setattr(
        mc,
        "parse_args",
        lambda: SimpleNamespace(
            num_trials=1,
            seed=0,
            plot=False,
            multi_noise=True,
            output_summary=output_summary,
            verbose=False,
        ),
    )
    monkeypatch.setattr(mc, "configure_logging", MagicMock())
    monkeypatch.setattr(mc, "run_single_scenario", MagicMock(return_value={"test": "ok"}))
    main()
    assert mc.run_single_scenario.call_count == 4  # noise_levels length 4
    # Index file should be created
    index_path = output_summary.with_suffix("").with_name(
        output_summary.with_suffix("").name + "_index.json"
    )
    assert index_path.exists()
