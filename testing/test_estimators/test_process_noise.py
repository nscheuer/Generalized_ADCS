import numpy as np

from ADCS.estimators.process_noise import (
    assemble_continuous_process_psd,
    error_state_transfer,
    van_loan_discretize,
)
from ADCS.satellite_hardware.errors import Bias
from ADCS.satellite_hardware.actuators import Actuator
from ADCS.satellite_hardware.disturbances import Disturbance
from ADCS.satellite_hardware.satellite import EstimatedSatellite
from ADCS.satellite_hardware.sensors import Sensor
from ADCS.state import EstimatorState


def test_estimator_state_exposes_full_and_tangent_block_layouts():
    state = EstimatorState(
        w=np.zeros(3),
        q=[1.0, 0.0, 0.0, 0.0],
        h=[1.0, 2.0],
        act_bias=[0.0],
        sens_bias=[0.0, 0.0],
        dist_param=[0.0],
    )

    assert state.full_slices == {
        "angular_velocity": slice(0, 3),
        "quaternion": slice(3, 7),
        "wheel_momentum": slice(7, 9),
        "actuator_bias": slice(9, 10),
        "sensor_bias": slice(10, 12),
        "disturbance_parameter": slice(12, 13),
    }
    assert state.tangent_slices["attitude"] == slice(3, 6)
    assert state.tangent_slices["wheel_momentum"] == slice(6, 8)
    assert state.tangent_slices["actuator_bias"] == slice(8, 9)
    assert state.tangent_slices["sensor_bias"] == slice(9, 11)
    assert state.tangent_slices["disturbance_parameter"] == slice(11, 12)
    assert state.tangent_slices["physical"] == slice(0, 8)


class _ParameterDisturbance(Disturbance):
    def __init__(self):
        super().__init__(estimate_dist=True, estimated_vector_length=1)
        self.std = np.array([0.4])
        self._main_param = np.zeros(1)

    @property
    def main_param(self):
        return self._main_param

    @main_param.setter
    def main_param(self, value):
        self._main_param = np.asarray(value, dtype=float)


def test_continuous_psd_uses_hardware_random_walk_rates_by_layout_block():
    satellite = EstimatedSatellite(
        actuators=[
            Actuator(
                axis=np.array([1.0, 0.0, 0.0]),
                u_max=1.0,
                bias=Bias(std_bias=0.2),
                estimate_bias=True,
            )
        ],
        sensors=[
            Sensor(output_length=1, bias=Bias(std_bias=0.3), estimate_bias=True)
        ],
        disturbances=[_ParameterDisturbance()],
    )
    state = EstimatorState(
        w=np.zeros(3),
        q=[1.0, 0.0, 0.0, 0.0],
        act_bias=[0.0],
        sens_bias=[0.0],
        dist_param=[0.0],
    )

    psd = assemble_continuous_process_psd(
        state,
        satellite,
        unmodeled_dynamics_psd=0.01,
        integration_error_psd=np.array([0.02] * state.tangent_size),
    )

    np.testing.assert_allclose(np.diag(psd)[:6], 0.03)
    np.testing.assert_allclose(np.diag(psd)[6:], [0.06, 0.11, 0.18])


class _LinearSatellite:
    def __init__(self):
        self._a = np.zeros((7, 7))
        self._a[0, 5] = 6.0

    def dynJacCore(self, state, control, orbital_state):
        return [
            self._a.T,
            np.empty((0, 7)),
            np.zeros((1, 7)),
            np.zeros((0, 7)),
            np.zeros((0, 7)),
        ]


def test_error_state_transfer_converts_jacobian_orientation_and_reduces_quaternion():
    state = EstimatorState(
        w=np.zeros(3),
        q=[1.0, 0.0, 0.0, 0.0],
        act_bias=[0.0],
    )
    transfer = error_state_transfer(state, _LinearSatellite(), np.empty(0), None)

    assert transfer.shape == (7, 7)
    assert transfer[0, 4] == 3.0
    np.testing.assert_allclose(transfer[4, 0], 0.0)


def test_van_loan_discretization_matches_random_walk_and_first_order_transition():
    transfer = np.array([[-2.0]])
    transition, covariance = van_loan_discretize(transfer, np.array([[3.0]]), 0.5)

    np.testing.assert_allclose(transition, [[np.exp(-1.0)]])
    np.testing.assert_allclose(covariance, [[3.0 * (1.0 - np.exp(-2.0)) / 4.0]])
