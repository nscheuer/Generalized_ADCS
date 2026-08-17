import numpy as np
import pytest

from ADCS.helpers.math_constants import MathConstants
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.disturbances import Dipole_Disturbance
from ADCS.satellite_hardware.errors import Bias, Noise
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.sensors import Gyro
from ADCS.state import EstimatorState, State


UNIT_VECTORS = MathConstants.unitvecs


def _make_estimated_satellite() -> EstimatedSatellite:
    return EstimatedSatellite(
        mass=4.0,
        J_0=np.diagflat([3.4, 2.9, 1.3]),
        actuators=[
            MTQ(
                axis=UNIT_VECTORS[index],
                max_torque=0.1,
                bias=Bias(bias=0.0, std_bias=1.0e-4),
                estimate_bias=True,
            )
            for index in range(3)
        ]
        + [
            RW(axis=UNIT_VECTORS[0], max_torque=1.0e-3, J=1.0e-4, h=0.0, h_max=0.1),
            RW(axis=UNIT_VECTORS[1], max_torque=1.0e-3, J=1.0e-4, h=0.0, h_max=0.1),
        ],
        sensors=[
            Gyro(
                axis=UNIT_VECTORS[index],
                bias=Bias(bias=0.0, std_bias=1.0e-4),
                noise=Noise(noise=0.0, std_noise=1.0e-4),
                estimate_bias=True,
            )
            for index in range(3)
        ],
        disturbances=[
            Dipole_Disturbance(dipole_torque=np.zeros(3), estimate_dist=True),
        ],
    )


def _state(
    satellite: EstimatedSatellite,
    *,
    h=None,
    act_bias=None,
    sens_bias=None,
    dist_param=None,
    full_cov: bool = False,
) -> EstimatorState:
    h = np.full(satellite.number_RW, 0.2) if h is None else np.asarray(h, dtype=float)
    act_bias = (
        np.full(satellite.act_bias_len, 1.0e-3)
        if act_bias is None
        else np.asarray(act_bias, dtype=float)
    )
    sens_bias = (
        np.full(satellite.att_sens_bias_len, -2.0e-3)
        if sens_bias is None
        else np.asarray(sens_bias, dtype=float)
    )
    dist_param = (
        np.full(satellite.dist_param_len, 3.0e-6)
        if dist_param is None
        else np.asarray(dist_param, dtype=float)
    )
    size = 7 + h.size + act_bias.size + sens_bias.size + dist_param.size
    cov_size = size if full_cov else size - 1
    return EstimatorState(
        w=np.zeros(3),
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        h=h,
        act_bias=act_bias,
        sens_bias=sens_bias,
        dist_param=dist_param,
        cov=np.eye(cov_size) * 1.0e-3,
        int_cov=np.eye(cov_size) * 1.0e-9,
    )


def test_match_estimate_rejects_non_estimator_state() -> None:
    satellite = _make_estimated_satellite()

    with pytest.raises(TypeError, match="est_state must be an EstimatorState"):
        satellite.match_estimate(
            State(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0], h=np.zeros(satellite.number_RW)),
            dt=1.0,
        )


def test_match_estimate_rejects_wrong_augmented_size() -> None:
    satellite = _make_estimated_satellite()
    est_state = _state(satellite, dist_param=np.zeros(satellite.dist_param_len + 1))

    with pytest.raises(ValueError, match="wrong size"):
        satellite.match_estimate(est_state, dt=1.0)


def test_match_estimate_rejects_wrong_wheel_state_count_after_size_check() -> None:
    satellite = _make_estimated_satellite()
    est_state = _state(
        satellite,
        h=np.zeros(satellite.number_RW - 1),
        act_bias=np.zeros(satellite.act_bias_len + 1),
    )

    with pytest.raises(ValueError, match="wheel states"):
        satellite.match_estimate(est_state, dt=1.0)


def test_match_estimate_rejects_wrong_actuator_bias_count_after_size_check() -> None:
    satellite = _make_estimated_satellite()
    est_state = _state(
        satellite,
        act_bias=np.zeros(satellite.act_bias_len - 1),
        sens_bias=np.zeros(satellite.att_sens_bias_len + 1),
    )

    with pytest.raises(ValueError, match="actuator biases"):
        satellite.match_estimate(est_state, dt=1.0)


def test_match_estimate_rejects_wrong_sensor_bias_count_after_size_check() -> None:
    satellite = _make_estimated_satellite()
    est_state = _state(
        satellite,
        sens_bias=np.zeros(satellite.att_sens_bias_len - 1),
        dist_param=np.zeros(satellite.dist_param_len + 1),
    )

    with pytest.raises(ValueError, match="sensor biases"):
        satellite.match_estimate(est_state, dt=1.0)


@pytest.mark.xfail(reason="dist_param length check is unreachable after augmented_size validation")
def test_match_estimate_rejects_wrong_disturbance_parameter_count_after_size_check() -> None:
    satellite = _make_estimated_satellite()
    est_state = _state(satellite, dist_param=np.zeros(satellite.dist_param_len - 1))

    with pytest.raises(ValueError, match="disturbance parameters"):
        satellite.match_estimate(est_state, dt=1.0)


def test_match_estimate_uses_full_size_integrated_covariance_without_index_shift() -> None:
    satellite = _make_estimated_satellite()
    expected_len = (
        satellite.state_len
        + satellite.act_bias_len
        + satellite.att_sens_bias_len
        + satellite.dist_param_len
    )
    est_state = _state(
        satellite,
        h=np.array([0.04, -0.03]),
        act_bias=np.array([1.0e-3, -2.0e-3, 3.0e-3]),
        sens_bias=np.array([4.0e-3, -5.0e-3, 6.0e-3]),
        dist_param=np.array([7.0e-6, -8.0e-6, 9.0e-6]),
        full_cov=True,
    )
    diagonal = np.arange(1, expected_len + 1, dtype=float) ** 2
    est_state.int_cov = np.diag(diagonal)

    satellite.match_estimate(est_state, dt=1.0)

    np.testing.assert_allclose([rw.h for rw in satellite.rw_actuators], est_state.h)
    np.testing.assert_allclose(
        [np.asarray(act.bias.bias).reshape(-1)[0] for act in satellite.actuators[:3]],
        est_state.act_bias,
    )
    np.testing.assert_allclose(
        [np.asarray(sensor.bias.bias).reshape(-1)[0] for sensor in satellite.attitude_sensors],
        est_state.sens_bias,
    )
    np.testing.assert_allclose(satellite.disturbances[0].main_param, est_state.dist_param)

    act_start = satellite.state_len
    sens_start = act_start + satellite.act_bias_len
    dist_start = sens_start + satellite.att_sens_bias_len
    np.testing.assert_allclose(
        [np.asarray(act.bias.std_bias).reshape(-1)[0] for act in satellite.actuators[:3]],
        np.sqrt(diagonal[act_start : act_start + satellite.act_bias_len]),
    )
    np.testing.assert_allclose(
        [np.asarray(sensor.bias.std_bias).reshape(-1)[0] for sensor in satellite.attitude_sensors],
        np.sqrt(diagonal[sens_start : sens_start + satellite.att_sens_bias_len]),
    )
    np.testing.assert_allclose(
        np.diag(satellite.disturbances[0].std),
        np.sqrt(diagonal[dist_start : dist_start + satellite.dist_param_len]),
    )
