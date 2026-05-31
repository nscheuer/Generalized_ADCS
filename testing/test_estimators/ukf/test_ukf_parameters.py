import numpy as np

from testing.test_estimators.ukf.helpers import make_baseline_sensors, make_orbital_state, make_satellites, make_ukf


def test_default_ukf_parameters_are_set():
    _, est_sat = make_satellites(sensors=make_baseline_sensors())
    ukf = make_ukf(est_sat, dt=7.5, cross_term=True, quat_as_vec=True)

    assert ukf.al == 1.0e-3
    assert ukf.bet == 2.0
    assert ukf.kap == 0.0
    assert ukf.vec_mode == 6
    assert ukf.dt == 7.5
    assert ukf.cross_term is True
    assert ukf.quat_as_vec is True


def test_weight_formulas_change_with_scaling_parameters():
    _, est_sat = make_satellites(sensors=make_baseline_sensors())
    ukf = make_ukf(est_sat)
    _, _, wts_before, _, _ = ukf.make_pts_and_wts(ukf.x_hat.val.copy(), [True] * len(est_sat.attitude_sensors))
    ukf.al = 0.2
    ukf.bet = 1.5
    ukf.kap = 2.0
    _, _, wts_after, _, _ = ukf.make_pts_and_wts(ukf.x_hat.val.copy(), [True] * len(est_sat.attitude_sensors))

    assert not np.allclose(wts_before, wts_after)


def test_quat_as_vec_changes_covariance_dimension_expectation():
    _, est_sat = make_satellites(sensors=make_baseline_sensors())
    reduced = make_ukf(est_sat, quat_as_vec=False)
    full = make_ukf(est_sat, quat_as_vec=True)

    assert reduced.x_hat.cov.shape[0] == reduced.x_hat.val.size - 1
    assert full.x_hat.cov.shape[0] == full.x_hat.val.size


def test_reset_updates_dt_and_process_covariance():
    _, est_sat = make_satellites(sensors=make_baseline_sensors())
    ukf = make_ukf(est_sat, dt=5.0)
    x_hat = ukf.x_hat.val.copy()
    P_hat = ukf.x_hat.cov.copy()
    Q_hat = ukf.x_hat.int_cov.copy() * 2.0

    ukf.reset(J2000=0.25, x_hat=x_hat, P_hat=P_hat, Q_hat=Q_hat, dt=12.0, cross_term=False)

    assert np.allclose(ukf.x_hat.int_cov, Q_hat)
    assert np.allclose(ukf.x_hat.val, x_hat)

