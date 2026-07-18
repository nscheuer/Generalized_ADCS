import numpy as np
import pytest

from ADCS.estimators.attitude_estimators import SRUAKF, UAKF
from ADCS.helpers.math_constants import MathConstants
from ADCS.satellite_hardware.actuators import MTQ
from ADCS.satellite_hardware.disturbances import Dipole_Disturbance, GG_Disturbance
from ADCS.satellite_hardware.errors import Bias, Noise
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.sensors import Gyro
from ADCS.state import EstimatedState, State


UNIT_VECTORS = MathConstants.unitvecs


def make_estimated_satellite() -> EstimatedSatellite:
    return EstimatedSatellite(
        mass=4.0,
        J_0=np.diagflat([3.4, 2.9, 1.3]),
        actuators=[MTQ(axis=UNIT_VECTORS[index], max_torque=0.1) for index in range(3)],
        sensors=[
            Gyro(
                axis=UNIT_VECTORS[index],
                bias=Bias(bias=0.0, std_bias=1e-4),
                noise=Noise(noise=0.0, std_noise=1e-4),
                estimate_bias=True,
            )
            for index in range(3)
        ],
        disturbances=[Dipole_Disturbance(dipole_torque=np.zeros(3), estimate_dist=True)],
    )


def build_filter(filter_type, estimated_satellite: EstimatedSatellite, x_hat: EstimatedState):
    reduced_length = (
        estimated_satellite.state_len
        - 1
        + estimated_satellite.act_bias_len
        + estimated_satellite.att_sens_bias_len
        + estimated_satellite.dist_param_len
    )
    return filter_type(
        est_sat=estimated_satellite,
        J2000=0.22,
        x_hat=x_hat,
        P_hat=np.eye(reduced_length) * 1e-3,
        Q_hat=np.eye(reduced_length) * 1e-9,
        dt=1.0,
        cross_term=True,
        quat_as_vec=False,
    )


def test_dipole_main_param_has_estimation_shape():
    disturbance = Dipole_Disturbance(dipole_torque=np.zeros(3), estimate_dist=True)
    assert disturbance.main_param.size == disturbance.estimated_vector_length == 3
    assert disturbance.std.shape == (3,)


def test_dipole_main_param_changes_torque_output():
    disturbance = Dipole_Disturbance(dipole_torque=np.zeros(3), estimate_dist=True)

    class StubOrbitalState:
        def get_state_vector(self, x):
            return {"b": np.array([1e-5, -2e-5, 3e-5])}

    state = State(w=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0]))
    zero_torque = np.asarray(disturbance.torque(state, StubOrbitalState()), dtype=float)
    disturbance.main_param = np.array([2.0e-4, -1.0e-4, 5.0e-5])
    driven_torque = np.asarray(disturbance.torque(state, StubOrbitalState()), dtype=float)

    np.testing.assert_allclose(zero_torque, 0.0, atol=0.0)
    assert np.linalg.norm(driven_torque) > 0.0
    np.testing.assert_allclose(disturbance.main_param, [2.0e-4, -1.0e-4, 5.0e-5])


def test_estimated_satellite_tracks_disturbance_parameter_length():
    estimated_satellite = make_estimated_satellite()
    assert estimated_satellite.dist_param_len == 3


@pytest.mark.parametrize("filter_type", [UAKF, SRUAKF])
def test_filter_builds_with_disturbance_parameter_augmented_state(filter_type):
    estimated_satellite = make_estimated_satellite()
    x_hat = EstimatedState(
        w=np.zeros(3),
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        sens_bias=np.zeros(3),
        dist_param=np.array([1.0e-6, -2.0e-6, 3.0e-6]),
    )
    filter_instance = build_filter(filter_type, estimated_satellite, x_hat)
    assert filter_instance is not None


@pytest.mark.parametrize("filter_type", [UAKF, SRUAKF])
def test_match_estimate_writes_disturbance_parameters(filter_type):
    estimated_satellite = make_estimated_satellite()
    expected = np.array([1.0e-6, -2.0e-6, 3.0e-6])
    x_hat = EstimatedState(
        w=np.zeros(3),
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        sens_bias=np.zeros(3),
        dist_param=expected,
    )

    build_filter(filter_type, estimated_satellite, x_hat)
    actual = np.asarray(estimated_satellite.disturbances[0].main_param, dtype=float).reshape(3)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)


def test_non_estimating_base_disturbance_main_param_fails_loudly():
    disturbance = GG_Disturbance()
    assert disturbance.active is True
    assert hasattr(disturbance, "std")
    with pytest.raises(NotImplementedError):
        _ = disturbance.main_param
