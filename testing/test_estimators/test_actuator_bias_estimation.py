import numpy as np
import pytest

from ADCS.estimators.old_attitude_estimators import SRUAKF, UAKF
from ADCS.helpers.math_constants import MathConstants
from ADCS.satellite_hardware.actuators import MTQ
from ADCS.satellite_hardware.errors import Bias, Noise
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.sensors import Gyro
from ADCS.state import EstimatorState


UNIT_VECTORS = MathConstants.unitvecs


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
