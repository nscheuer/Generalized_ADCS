import numpy as np
import pytest

from ADCS.estimators.estimator_helpers.estimator_helpers import EstimatedArray
from testing.test_estimators.ukf.helpers import (
    make_baseline_sensors,
    make_mtqs,
    make_orbital_state,
    make_satellites,
    make_state,
    make_ukf,
)


def test_uakf_constructor_validates_reduced_covariance_shape():
    _, est_sat = make_satellites(sensors=make_baseline_sensors())
    x_hat = np.concatenate([make_state(), np.zeros(est_sat.act_bias_len + est_sat.att_sens_bias_len + est_sat.dist_param_len)])

    with pytest.raises(ValueError):
        make_ukf(est_sat, x_hat=x_hat, P_hat=np.eye(x_hat.size), Q_hat=np.eye(x_hat.size), quat_as_vec=False)


def test_uakf_constructor_validates_full_quaternion_covariance_shape():
    _, est_sat = make_satellites(sensors=make_baseline_sensors())
    x_hat = np.concatenate([make_state(), np.zeros(est_sat.act_bias_len + est_sat.att_sens_bias_len + est_sat.dist_param_len)])

    with pytest.raises(ValueError):
        make_ukf(est_sat, x_hat=x_hat, P_hat=np.eye(x_hat.size - 1), Q_hat=np.eye(x_hat.size - 1), quat_as_vec=True)


def test_determine_covariances_to_use_tracks_nonzero_control_covariance():
    _, est_sat = make_satellites(sensors=make_baseline_sensors(), actuators=make_mtqs(std_noise=1.0e-5), estimated_actuators=make_mtqs(std_noise=1.0e-5))
    ukf = make_ukf(est_sat)
    state_cov = ukf.x_hat.cov.copy()
    sens_cov = np.eye(3)
    control_cov = est_sat.control_cov()
    int_cov = ukf.x_hat.int_cov.copy()

    include = ukf.determine_covariances_to_use(state_cov, sens_cov, control_cov, int_cov)

    assert include == [True, False, True, False]


def test_determine_covariances_to_use_ignores_zero_control_covariance():
    zero_noise_acts = make_mtqs(std_noise=0.0)
    _, est_sat = make_satellites(sensors=make_baseline_sensors(), actuators=zero_noise_acts, estimated_actuators=zero_noise_acts)
    ukf = make_ukf(est_sat)

    include = ukf.determine_covariances_to_use(ukf.x_hat.cov, np.eye(3), est_sat.control_cov(), ukf.x_hat.int_cov)

    assert include == [True, False, False, False]


def test_sat_match_updates_reaction_wheel_momentum_and_bias_state():
    sensors = make_baseline_sensors(estimate_gyro_bias=True)
    _, est_sat = make_satellites(
        sensors=sensors,
        estimated_sensors=sensors,
        actuators=make_mtqs() + [],
        estimated_actuators=make_mtqs() + [],
    )
    ukf = make_ukf(est_sat)
    state = ukf.x_hat.val.copy()
    state[7:10] = np.array([1.0e-3, -2.0e-3, 3.0e-3])

    ukf.sat_match(est_sat, state)

    gyro_biases = np.array([sensor.bias.bias.item() for sensor in est_sat.attitude_sensors if hasattr(sensor, "bias") and sensor.bias])
    assert np.allclose(gyro_biases[:3], np.array([1.0e-3, -2.0e-3, 3.0e-3]))


def test_update_returns_normalized_quaternion_and_symmetric_covariance():
    real_sat, est_sat = make_satellites(sensors=make_baseline_sensors(), estimated_sensors=make_baseline_sensors())
    ukf = make_ukf(est_sat)
    x_true = make_state(q=np.array([0.91, 0.22, -0.11, 0.33]))
    os = make_orbital_state()
    sensors = real_sat.noiseless_sensor_readings(x_true, os)

    out = ukf.update(u=np.zeros(len(real_sat.actuators)), sensors=sensors, os=os)

    assert np.isclose(np.linalg.norm(out[3:7]), 1.0)
    assert np.allclose(ukf.x_hat.cov, ukf.x_hat.cov.T)


def test_cross_term_false_zeros_bias_cross_blocks(monkeypatch):
    sensors = make_baseline_sensors(estimate_gyro_bias=True)
    acts = make_mtqs(estimate_bias=True)
    _, est_sat = make_satellites(sensors=sensors, estimated_sensors=sensors, actuators=acts, estimated_actuators=acts)
    ukf = make_ukf(est_sat, cross_term=False)
    size = ukf.x_hat.cov.shape[0]
    cov = np.ones((size, size))
    cov = 0.5 * (cov + cov.T)
    fake = EstimatedArray(val=ukf.x_hat.val.copy(), cov=cov, int_cov=ukf.x_hat.int_cov.copy())

    monkeypatch.setattr(ukf, "update_core", lambda u, sensors, os: fake)
    ukf.update(u=np.zeros(len(acts)), sensors=np.zeros(sum(s.output_length for s in est_sat.attitude_sensors)), os=make_orbital_state())

    ab0 = est_sat.state_len - 1
    ab1 = ab0 + est_sat.act_bias_len
    sb0 = ab1
    sb1 = sb0 + est_sat.att_sens_bias_len
    assert np.allclose(ukf.x_hat.cov[ab0:ab1, sb0:sb1], 0.0)
    assert np.allclose(ukf.x_hat.cov[sb0:sb1, ab0:ab1], 0.0)


def test_cross_term_true_preserves_bias_cross_blocks(monkeypatch):
    sensors = make_baseline_sensors(estimate_gyro_bias=True)
    acts = make_mtqs(estimate_bias=True)
    _, est_sat = make_satellites(sensors=sensors, estimated_sensors=sensors, actuators=acts, estimated_actuators=acts)
    ukf = make_ukf(est_sat, cross_term=True)
    size = ukf.x_hat.cov.shape[0]
    cov = np.ones((size, size))
    cov = 0.5 * (cov + cov.T)
    fake = EstimatedArray(val=ukf.x_hat.val.copy(), cov=cov, int_cov=ukf.x_hat.int_cov.copy())

    monkeypatch.setattr(ukf, "update_core", lambda u, sensors, os: fake)
    ukf.update(u=np.zeros(len(acts)), sensors=np.zeros(sum(s.output_length for s in est_sat.attitude_sensors)), os=make_orbital_state())

    ab0 = est_sat.state_len - 1
    ab1 = ab0 + est_sat.act_bias_len
    sb0 = ab1
    sb1 = sb0 + est_sat.att_sens_bias_len
    assert np.allclose(ukf.x_hat.cov[ab0:ab1, sb0:sb1], 1.0)
