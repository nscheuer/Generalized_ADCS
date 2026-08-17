import numpy as np

from testing.test_estimators.srukf.helpers import make_baseline_sensors, make_satellites, make_srukf


def test_default_srukf_parameters_and_square_roots_are_set():
    _, est_sat = make_satellites(sensors=make_baseline_sensors())
    srukf = make_srukf(est_sat, dt=7.5, cross_term=True, quat_as_vec=True)
    assert srukf.al == 1.0e-3
    assert srukf.bet == 2.0
    assert srukf.kap == 0.0
    assert srukf.vec_mode == 6
    assert srukf.dt == 7.5
    assert srukf.cross_term is True
    assert srukf.quat_as_vec is True
    assert srukf.S.shape == srukf.x_hat.cov.shape
    assert srukf.S_Q.shape == srukf.x_hat.int_cov.shape


def test_weight_formulas_change_with_scaling_parameters():
    _, est_sat = make_satellites(sensors=make_baseline_sensors())
    srukf = make_srukf(est_sat)
    _, _, wts_before, _, _ = srukf.make_pts_and_wts(srukf.x_hat.as_estimator_array(), [True] * len(est_sat.attitude_sensors))
    srukf.al = 0.2
    srukf.bet = 1.5
    srukf.kap = 2.0
    _, _, wts_after, _, _ = srukf.make_pts_and_wts(srukf.x_hat.as_estimator_array(), [True] * len(est_sat.attitude_sensors))
    assert not np.allclose(wts_before, wts_after)


def test_quat_as_vec_changes_covariance_dimension_expectation():
    _, est_sat = make_satellites(sensors=make_baseline_sensors())
    reduced = make_srukf(est_sat, quat_as_vec=False)
    full = make_srukf(est_sat, quat_as_vec=True)
    assert reduced.x_hat.cov.shape[0] == reduced.x_hat.augmented_size - 1
    assert full.x_hat.cov.shape[0] == full.x_hat.augmented_size


def test_zero_covariance_initializes_zero_square_root():
    _, est_sat = make_satellites(sensors=make_baseline_sensors())
    probe = make_srukf(est_sat)
    zero = np.zeros_like(probe.x_hat.cov)
    srukf = make_srukf(est_sat, P_hat=zero, Q_hat=zero)
    assert np.allclose(srukf.S, 0.0)
    assert np.allclose(srukf.S_Q, 0.0)
