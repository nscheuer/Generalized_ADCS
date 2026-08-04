import numpy as np

from testing.test_estimators.srukf.helpers import make_baseline_sensors, make_mtqs, make_satellites, make_srukf


def test_make_pts_and_wts_returns_expected_counts_and_weight_sums():
    _, est_sat = make_satellites(sensors=make_baseline_sensors(), actuators=make_mtqs(), estimated_actuators=make_mtqs())
    srukf = make_srukf(est_sat)
    L, pts, wts_m, wts_c, sig0 = srukf.make_pts_and_wts(srukf.x_hat.val.copy(), [True] * len(est_sat.attitude_sensors))
    assert len(pts) == 2 * L + 1
    assert wts_m.shape == (2 * L + 1,)
    assert wts_c.shape == (2 * L + 1,)
    assert sig0.shape == (2 * L + 1, srukf.x_hat.val.size)
    assert np.isclose(np.sum(wts_m), 1.0)


def test_make_pts_and_wts_augmented_dimension_includes_control_noise():
    _, est_sat = make_satellites(sensors=make_baseline_sensors(), actuators=make_mtqs(std_noise=1.0e-5), estimated_actuators=make_mtqs(std_noise=1.0e-5))
    srukf = make_srukf(est_sat)
    L, _, _, _, _ = srukf.make_pts_and_wts(srukf.x_hat.val.copy(), [True] * len(est_sat.attitude_sensors))
    expected = srukf.S.shape[0] + est_sat.control_cov().shape[0]
    assert L == expected


def test_make_pts_and_wts_center_weight_matches_formula():
    _, est_sat = make_satellites(sensors=make_baseline_sensors(), actuators=make_mtqs(), estimated_actuators=make_mtqs())
    srukf = make_srukf(est_sat)
    L, _, wts_m, wts_c, _ = srukf.make_pts_and_wts(srukf.x_hat.val.copy(), [True] * len(est_sat.attitude_sensors))
    lam = srukf.al**2 * (srukf.kap + L) - L
    denom = L + lam
    assert np.isclose(wts_m[0], lam / denom)
    assert np.isclose(wts_c[0], lam / denom + (1.0 - srukf.al**2 + srukf.bet))


def test_make_pts_and_wts_returns_control_only_offsets_for_non_state_block():
    _, est_sat = make_satellites(sensors=make_baseline_sensors(), actuators=make_mtqs(std_noise=1.0e-5), estimated_actuators=make_mtqs(std_noise=1.0e-5))
    srukf = make_srukf(est_sat)
    _, pts, _, _, _ = srukf.make_pts_and_wts(srukf.x_hat.val.copy(), [True] * len(est_sat.attitude_sensors))
    state_sigma_count = 2 * srukf.S.shape[0]
    first_control_point = pts[1 + state_sigma_count]
    assert np.allclose(first_control_point[0], srukf.x_hat.val)
    assert np.any(first_control_point[2] != 0.0)
    assert np.allclose(first_control_point[1], 0.0)
    assert np.allclose(first_control_point[3], 0.0)


def test_make_pts_and_wts_keeps_quaternions_normalized_in_error_state_mode():
    _, est_sat = make_satellites(sensors=make_baseline_sensors())
    srukf = make_srukf(est_sat, quat_as_vec=False)
    _, pts, _, _, sig0 = srukf.make_pts_and_wts(srukf.x_hat.val.copy(), [True] * len(est_sat.attitude_sensors))
    assert np.allclose(np.linalg.norm(sig0[:, 3:7], axis=1), 1.0)
    assert np.allclose([np.linalg.norm(point[0][3:7]) for point in pts], 1.0)


def test_make_pts_and_wts_keeps_quaternions_normalized_in_full_quaternion_mode():
    _, est_sat = make_satellites(sensors=make_baseline_sensors())
    srukf = make_srukf(est_sat, quat_as_vec=True)
    _, pts, _, _, sig0 = srukf.make_pts_and_wts(srukf.x_hat.val.copy(), [True] * len(est_sat.attitude_sensors))
    assert np.allclose(np.linalg.norm(sig0[:, 3:7], axis=1), 1.0)
    assert np.allclose([np.linalg.norm(point[0][3:7]) for point in pts], 1.0)


def test_make_pts_and_wts_zero_covariance_returns_mean_repeated():
    _, est_sat = make_satellites(sensors=make_baseline_sensors(), actuators=make_mtqs(std_noise=0.0), estimated_actuators=make_mtqs(std_noise=0.0))
    zero = np.zeros_like(make_srukf(est_sat).x_hat.cov)
    srukf = make_srukf(est_sat, P_hat=zero, Q_hat=zero)
    L, pts, _, _, sig0 = srukf.make_pts_and_wts(srukf.x_hat.val.copy(), [True] * len(est_sat.attitude_sensors))
    assert L == srukf.S.shape[0] + est_sat.control_cov().shape[0]
    assert np.allclose(sig0, np.repeat(srukf.x_hat.val[None, :], 2 * L + 1, axis=0))
    state_rows = np.asarray([p[0] for p in pts])
    assert np.allclose(state_rows, np.repeat(srukf.x_hat.val[None, :], len(pts), axis=0))
