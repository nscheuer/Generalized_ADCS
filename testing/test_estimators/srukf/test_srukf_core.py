import numpy as np
import pytest

from ADCS.estimators.estimator_helpers.estimator_helpers import EstimatedArray
from testing.test_estimators.srukf.helpers import (
    make_baseline_sensors,
    make_mtqs,
    make_orbital_state,
    make_satellites,
    make_state,
    make_srukf,
)


def test_srukf_constructor_validates_reduced_covariance_shape():
    _, est_sat = make_satellites(sensors=make_baseline_sensors())
    x_hat = np.concatenate([make_state(), np.zeros(est_sat.act_bias_len + est_sat.att_sens_bias_len + est_sat.dist_param_len)])
    with pytest.raises(ValueError):
        make_srukf(est_sat, x_hat=x_hat, P_hat=np.eye(x_hat.size), Q_hat=np.eye(x_hat.size), quat_as_vec=False)


def test_srukf_constructor_validates_full_quaternion_covariance_shape():
    _, est_sat = make_satellites(sensors=make_baseline_sensors())
    x_hat = np.concatenate([make_state(), np.zeros(est_sat.act_bias_len + est_sat.att_sens_bias_len + est_sat.dist_param_len)])
    with pytest.raises(ValueError):
        make_srukf(est_sat, x_hat=x_hat, P_hat=np.eye(x_hat.size - 1), Q_hat=np.eye(x_hat.size - 1), quat_as_vec=True)


def test_srukf_determine_covariances_to_use_tracks_nonzero_control_covariance():
    _, est_sat = make_satellites(sensors=make_baseline_sensors(), actuators=make_mtqs(std_noise=1.0e-5), estimated_actuators=make_mtqs(std_noise=1.0e-5))
    srukf = make_srukf(est_sat)
    include = srukf.determine_covariances_to_use(srukf.x_hat.cov.copy(), np.eye(3), est_sat.control_cov(), srukf.x_hat.int_cov.copy())
    assert include == [True, False, True, False]


def test_srukf_sat_match_updates_bias_state():
    sensors = make_baseline_sensors(estimate_gyro_bias=True)
    _, est_sat = make_satellites(sensors=sensors, estimated_sensors=sensors, actuators=make_mtqs(), estimated_actuators=make_mtqs())
    srukf = make_srukf(est_sat)
    state = srukf.x_hat.val.copy()
    state[7:10] = np.array([1.0e-3, -2.0e-3, 3.0e-3])
    srukf.sat_match(est_sat, state)
    gyro_biases = np.array([sensor.bias.bias.item() for sensor in est_sat.attitude_sensors if hasattr(sensor, "bias") and sensor.bias])
    assert np.allclose(gyro_biases[:3], np.array([1.0e-3, -2.0e-3, 3.0e-3]))


def test_srukf_update_returns_normalized_quaternion_symmetric_covariance_and_square_root():
    real_sat, est_sat = make_satellites(sensors=make_baseline_sensors(), estimated_sensors=make_baseline_sensors())
    srukf = make_srukf(est_sat)
    x_true = make_state(q=np.array([0.91, 0.22, -0.11, 0.33]))
    os = make_orbital_state()
    sensors = real_sat.noiseless_sensor_readings(x_true, os)
    out = srukf.update(u=np.zeros(len(real_sat.actuators)), sensors=sensors, os=os)
    assert np.isclose(np.linalg.norm(out[3:7]), 1.0)
    assert np.allclose(srukf.x_hat.cov, srukf.x_hat.cov.T)
    assert srukf.S.shape == srukf.x_hat.cov.shape


def test_srukf_cross_term_false_zeros_bias_cross_blocks(monkeypatch):
    sensors = make_baseline_sensors(estimate_gyro_bias=True)
    acts = make_mtqs(estimate_bias=True)
    _, est_sat = make_satellites(sensors=sensors, estimated_sensors=sensors, actuators=acts, estimated_actuators=acts)
    srukf = make_srukf(est_sat, cross_term=False)
    size = srukf.x_hat.cov.shape[0]
    cov = np.ones((size, size))
    cov = 0.5 * (cov + cov.T)
    fake = EstimatedArray(val=srukf.x_hat.val.copy(), cov=cov, int_cov=srukf.x_hat.int_cov.copy())
    monkeypatch.setattr(srukf, "update_core", lambda u, sensors, os: fake)
    srukf.update(u=np.zeros(len(acts)), sensors=np.zeros(sum(s.output_length for s in est_sat.attitude_sensors)), os=make_orbital_state())
    ab0 = est_sat.state_len - 1
    ab1 = ab0 + est_sat.act_bias_len
    sb0 = ab1
    sb1 = sb0 + est_sat.att_sens_bias_len
    assert np.allclose(srukf.x_hat.cov[ab0:ab1, sb0:sb1], 0.0)


def test_srukf_cross_term_true_preserves_bias_cross_blocks(monkeypatch):
    sensors = make_baseline_sensors(estimate_gyro_bias=True)
    acts = make_mtqs(estimate_bias=True)
    _, est_sat = make_satellites(sensors=sensors, estimated_sensors=sensors, actuators=acts, estimated_actuators=acts)
    srukf = make_srukf(est_sat, cross_term=True)
    size = srukf.x_hat.cov.shape[0]
    cov = np.ones((size, size))
    cov = 0.5 * (cov + cov.T)
    fake = EstimatedArray(val=srukf.x_hat.val.copy(), cov=cov, int_cov=srukf.x_hat.int_cov.copy())
    monkeypatch.setattr(srukf, "update_core", lambda u, sensors, os: fake)
    srukf.update(u=np.zeros(len(acts)), sensors=np.zeros(sum(s.output_length for s in est_sat.attitude_sensors)), os=make_orbital_state())
    ab0 = est_sat.state_len - 1
    ab1 = ab0 + est_sat.act_bias_len
    sb0 = ab1
    sb1 = sb0 + est_sat.att_sens_bias_len
    assert np.allclose(srukf.x_hat.cov[ab0:ab1, sb0:sb1], 1.0)
