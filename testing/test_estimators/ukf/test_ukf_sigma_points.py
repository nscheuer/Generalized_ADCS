import numpy as np
import pytest

from testing.test_estimators.ukf.helpers import make_baseline_sensors, make_mtqs, make_satellites, make_ukf


def test_make_pts_and_wts_returns_expected_counts_and_weight_sums():
    _, est_sat = make_satellites(sensors=make_baseline_sensors(), actuators=make_mtqs(), estimated_actuators=make_mtqs())
    ukf = make_ukf(est_sat)

    L, pts, wts_m, wts_c, sig0 = ukf.make_pts_and_wts(ukf.x_hat.as_estimator_array(), [True] * len(est_sat.attitude_sensors))

    assert len(pts) == 2 * L + 1
    assert wts_m.shape == (2 * L + 1,)
    assert wts_c.shape == (2 * L + 1,)
    assert sig0.shape == (2 * L + 1, ukf.x_hat.augmented_size)
    assert np.isclose(np.sum(wts_m), 1.0)


def test_make_pts_and_wts_augmented_dimension_includes_control_noise():
    _, est_sat = make_satellites(sensors=make_baseline_sensors(), actuators=make_mtqs(std_noise=1.0e-5), estimated_actuators=make_mtqs(std_noise=1.0e-5))
    ukf = make_ukf(est_sat)

    L, _, _, _, _ = ukf.make_pts_and_wts(ukf.x_hat.as_estimator_array(), [True] * len(est_sat.attitude_sensors))

    expected = ukf.x_hat.cov.shape[0] + est_sat.control_cov().shape[0]
    assert L == expected


def test_make_pts_and_wts_center_weight_matches_formula():
    _, est_sat = make_satellites(sensors=make_baseline_sensors(), actuators=make_mtqs(), estimated_actuators=make_mtqs())
    ukf = make_ukf(est_sat)

    L, _, wts_m, wts_c, _ = ukf.make_pts_and_wts(ukf.x_hat.as_estimator_array(), [True] * len(est_sat.attitude_sensors))
    lam = ukf.al**2 * (ukf.kap + L) - L
    denom = L + lam

    assert np.isclose(wts_m[0], lam / denom)
    assert np.isclose(wts_c[0], lam / denom + (1.0 - ukf.al**2 + ukf.bet))


def test_make_pts_and_wts_returns_control_only_offsets_for_non_state_block():
    _, est_sat = make_satellites(sensors=make_baseline_sensors(), actuators=make_mtqs(std_noise=1.0e-5), estimated_actuators=make_mtqs(std_noise=1.0e-5))
    ukf = make_ukf(est_sat)

    L, pts, _, _, _ = ukf.make_pts_and_wts(ukf.x_hat.as_estimator_array(), [True] * len(est_sat.attitude_sensors))
    state_sigma_count = 2 * ukf.x_hat.cov.shape[0]
    first_control_point = pts[1 + state_sigma_count]

    assert np.allclose(first_control_point[0], ukf.x_hat.as_estimator_array())
    assert np.any(first_control_point[2] != 0.0)
    assert np.allclose(first_control_point[1], 0.0)
    assert np.allclose(first_control_point[3], 0.0)


def test_make_pts_and_wts_keeps_quaternions_normalized_in_error_state_mode():
    _, est_sat = make_satellites(sensors=make_baseline_sensors())
    ukf = make_ukf(est_sat, quat_as_vec=False)

    _, pts, _, _, sig0 = ukf.make_pts_and_wts(ukf.x_hat.as_estimator_array(), [True] * len(est_sat.attitude_sensors))

    assert np.allclose(np.linalg.norm(sig0[:, 3:7], axis=1), 1.0)
    assert np.allclose([np.linalg.norm(point[0][3:7]) for point in pts], 1.0)


def test_make_pts_and_wts_keeps_quaternions_normalized_in_full_quaternion_mode():
    _, est_sat = make_satellites(sensors=make_baseline_sensors())
    ukf = make_ukf(est_sat, quat_as_vec=True)

    _, pts, _, _, sig0 = ukf.make_pts_and_wts(ukf.x_hat.as_estimator_array(), [True] * len(est_sat.attitude_sensors))

    assert np.allclose(np.linalg.norm(sig0[:, 3:7], axis=1), 1.0)
    assert np.allclose([np.linalg.norm(point[0][3:7]) for point in pts], 1.0)


def test_make_pts_and_wts_zero_covariance_raises():
    _, est_sat = make_satellites(sensors=make_baseline_sensors(), actuators=make_mtqs(std_noise=0.0), estimated_actuators=make_mtqs(std_noise=0.0))
    zero = np.zeros_like(make_ukf(est_sat).x_hat.cov)
    ukf = make_ukf(est_sat, P_hat=zero, Q_hat=zero)

    with pytest.raises(np.linalg.LinAlgError):
        ukf.make_pts_and_wts(ukf.x_hat.as_estimator_array(), [True] * len(est_sat.attitude_sensors))
