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


def test_make_pts_and_wts_zero_control_cov_is_excluded_and_does_not_raise():
    # Zero actuator noise must drop the control block from the augmented
    # dimension entirely (previously L counted it while no points were
    # generated, mismatching the weight vector length).
    _, est_sat = make_satellites(sensors=make_baseline_sensors(), actuators=make_mtqs(std_noise=0.0), estimated_actuators=make_mtqs(std_noise=0.0))
    ukf = make_ukf(est_sat)

    L, pts, wts_m, wts_c, sig0 = ukf.make_pts_and_wts(ukf.x_hat.as_estimator_array(), [True] * len(est_sat.attitude_sensors))

    assert L == ukf.x_hat.cov.shape[0]
    assert len(pts) == 2 * L + 1 == wts_m.size == wts_c.size
    assert all(np.allclose(p[2], 0.0) for p in pts)


def test_make_pts_and_wts_non_pd_control_cov_uses_eigh_fallback():
    # A PSD-singular control covariance (e.g. rank-deficient actuator noise)
    # fails Cholesky; the eigh square-root fallback must keep the update alive
    # with the full 2L+1 point set instead of raising.
    _, est_sat = make_satellites(sensors=make_baseline_sensors(), actuators=make_mtqs(), estimated_actuators=make_mtqs())
    ukf = make_ukf(est_sat)
    v = np.array([1.0, 2.0, 3.0])
    singular = np.outer(v, v) * 1.0e-8  # rank 1: PSD but not PD
    est_sat.control_cov = lambda: singular

    L, pts, wts_m, wts_c, _ = ukf.make_pts_and_wts(ukf.x_hat.as_estimator_array(), [True] * len(est_sat.attitude_sensors))

    assert L == ukf.x_hat.cov.shape[0] + 3
    assert len(pts) == 2 * L + 1 == wts_m.size
    control_rows = np.asarray([p[2] for p in pts])
    assert np.all(np.isfinite(control_rows))
    assert np.any(np.abs(control_rows) > 0.0)
    # The unscented transform must realize the requested covariance from the
    # eigh-based square root: sum_i w_c,i * off_i off_i^T == control_cov.
    realized = sum(w * np.outer(row, row) for w, row in zip(wts_c[1:], control_rows[1:]))
    assert np.allclose(realized, singular, rtol=1e-9, atol=1e-20)
