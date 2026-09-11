import numpy as np
import pytest

from ADCS.estimators.old_attitude_estimators import SRUAKF, UAKF
from ADCS.helpers.math_constants import MathConstants
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.actuators import MTQ
from ADCS.satellite_hardware.errors import Bias, Noise
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.sensors import Gyro
from ADCS.state import EstimatorState


UNIT_VECTORS = MathConstants.unitvecs


def make_orbital_state() -> Orbital_State:
    return Orbital_State(
        ephem=Ephemeris(),
        J2000=0.22,
        R=np.array([7000.0, 0.0, 0.0]),
        V=np.array([0.0, 8.0, 0.0]),
        B=np.array([1.0e-5, 0.0, 0.0]),
        fast=True,
    )


def make_estimated_satellite() -> EstimatedSatellite:
    return EstimatedSatellite(
        mass=4.0,
        J_0=np.diagflat([3.4, 2.9, 1.3]),
        actuators=[
            MTQ(
                axis=UNIT_VECTORS[index],
                max_torque=0.1,
                bias=Bias(bias=0.0, std_bias=1e-4),
                estimate_bias=True,
            )
            for index in range(3)
        ],
        sensors=[
            Gyro(
                axis=UNIT_VECTORS[index],
                bias=Bias(bias=0.0, std_bias=1e-4),
                noise=Noise(noise=0.0, std_noise=1e-4),
            )
            for index in range(3)
        ],
    )


def build_filter(filter_type, estimated_satellite: EstimatedSatellite, x_hat: EstimatorState):
    reduced_length = (
        estimated_satellite.state_len
        - 1
        + estimated_satellite.act_bias_len
        + estimated_satellite.att_sens_bias_len
        + estimated_satellite.dist_param_len
    )
    covariance = np.eye(reduced_length) * 1e-3
    process_noise = np.eye(reduced_length) * 1e-9
    return filter_type(
        est_sat=estimated_satellite,
        J2000=0.22,
        x_hat=x_hat,
        P_hat=covariance,
        Q_hat=process_noise,
        dt=1.0,
        cross_term=True,
        quat_as_vec=False,
    )


def test_estimate_bias_creates_default_bias_object_for_mtq():
    actuator = MTQ(axis=UNIT_VECTORS[0], max_torque=0.1, estimate_bias=True)
    assert actuator.estimate_bias is True
    assert actuator.bias is not None
    actuator.bias.bias = np.array([0.01] * actuator.input_len)
    actuator.bias.std_bias = np.eye(actuator.input_len) * 1e-4


def test_estimated_satellite_tracks_actuator_bias_layout():
    estimated_satellite = make_estimated_satellite()
    assert estimated_satellite.act_bias_len == 3
    assert estimated_satellite.att_sens_bias_len == 0


@pytest.mark.parametrize("filter_type", [UAKF, SRUAKF])
def test_filter_builds_with_actuator_bias_augmented_state(filter_type):
    estimated_satellite = make_estimated_satellite()
    x_hat = EstimatorState(w=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0]), act_bias=np.zeros(3))
    filter_instance = build_filter(filter_type, estimated_satellite, x_hat)
    assert filter_instance is not None


@pytest.mark.parametrize("filter_type", [UAKF, SRUAKF])
def test_match_estimate_writes_actuator_bias_values(filter_type):
    estimated_satellite = make_estimated_satellite()
    expected_biases = np.array([2.0e-3, -1.0e-3, 3.0e-3])
    x_hat = EstimatorState(w=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0]), act_bias=expected_biases)

    build_filter(filter_type, estimated_satellite, x_hat)
    actual_biases = np.concatenate([np.atleast_1d(actuator.bias.bias) for actuator in estimated_satellite.actuators])
    np.testing.assert_allclose(actual_biases, expected_biases, rtol=0, atol=0)


@pytest.mark.parametrize("bias_value", [0.0, 0.01])
@pytest.mark.parametrize("std_bias", [0.0, 1e-4])
@pytest.mark.parametrize("estimate_bias", [False, True])
def test_mtq_bias_derivative_shapes_follow_estimate_bias(
    bias_value, std_bias, estimate_bias
):
    actuator = MTQ(
        axis=UNIT_VECTORS[0],
        max_torque=0.1,
        bias=Bias(bias=bias_value, std_bias=std_bias),
        estimate_bias=estimate_bias,
    )
    state = EstimatorState(w=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0]))
    orbital_state = make_orbital_state()
    expected_bias_len = 1 if estimate_bias or actuator.bias else 0

    assert actuator.dtorq__dbias(0.02, state, orbital_state).shape == (
        expected_bias_len,
        3,
    )
    assert actuator.ddtorq__dudbias(0.02, state, orbital_state).shape == (
        1,
        expected_bias_len,
        3,
    )
    assert actuator.ddtorq__dbiasdbias(0.02, state, orbital_state).shape == (
        expected_bias_len,
        expected_bias_len,
        3,
    )
    assert actuator.ddtorq__dbiasdbasestate(0.02, state, orbital_state).shape == (
        expected_bias_len,
        7,
        3,
    )


@pytest.mark.parametrize("std_bias", [0.0, 1e-4])
def test_dyn_jac_core_accepts_estimated_zero_actuator_bias(std_bias):
    actuator = MTQ(
        axis=UNIT_VECTORS[0],
        max_torque=0.1,
        bias=Bias(bias=0.0, std_bias=std_bias),
        estimate_bias=True,
    )
    satellite = EstimatedSatellite(
        mass=4.0,
        J_0=np.diagflat([3.4, 2.9, 1.3]),
        actuators=[actuator],
    )
    state = EstimatorState(
        w=np.zeros(3),
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        act_bias=np.zeros(1),
    )

    dxdot__dab = satellite.dynJacCore(
        state,
        np.array([0.02]),
        make_orbital_state(),
    )[2]

    assert satellite.act_bias_len == 1
    assert dxdot__dab.shape == (1, satellite.state_len)
