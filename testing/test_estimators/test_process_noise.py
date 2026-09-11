import numpy as np

from ADCS.estimators.process_noise import (
    assemble_continuous_process_psd,
    discretize_process_noise,
    error_state_transfer,
    van_loan_discretize,
)
from ADCS.helpers.math_constants import MathConstants
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.errors import Bias, Noise
from ADCS.satellite_hardware.actuators import Actuator, RW
from ADCS.satellite_hardware.disturbances import Disturbance
from ADCS.satellite_hardware.satellite import EstimatedSatellite
from ADCS.satellite_hardware.sensors import Gyro, Sensor
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
        super().__init__(
            estimate_dist=True,
            estimated_vector_length=1,
            parameter_std_rate=0.4,
        )
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
    )

    np.testing.assert_allclose(np.diag(psd)[:6], 0.01)
    np.testing.assert_allclose(np.diag(psd)[6:], [0.04, 0.09, 0.16])


def test_continuous_psd_matches_full_quaternion_coordinates():
    satellite = EstimatedSatellite()
    state = EstimatorState(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0])

    psd = assemble_continuous_process_psd(
        state,
        satellite,
        unmodeled_dynamics_psd=0.2,
        quaternion_mode="full_quaternion",
    )

    assert psd.shape == (state.full_size, state.full_size)
    # Quaternion normalization removes radial noise at the identity attitude.
    np.testing.assert_allclose(np.diag(psd), [0.2, 0.2, 0.2, 0.0, 0.2, 0.2, 0.2])


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


def test_error_state_transfer_supports_full_and_tangent_coordinate_pairs():
    state = EstimatorState(
        w=np.array([0.01, -0.02, 0.03]),
        q=np.array([0.9, 0.2, -0.3, 0.1]) / np.linalg.norm([0.9, 0.2, -0.3, 0.1]),
        act_bias=[0.0],
    )
    modes = {
        "quaternion_vector": state.tangent_size,
        "full_quaternion": state.full_size,
    }

    for source_mode, source_size in modes.items():
        for target_mode, target_size in modes.items():
            transfer = error_state_transfer(
                state,
                _LinearSatellite(),
                np.empty(0),
                None,
                source_quaternion_mode=source_mode,
                target_quaternion_mode=target_mode,
            )

            assert transfer.shape == (target_size, source_size)


def test_van_loan_discretization_matches_random_walk_and_first_order_transition():
    transfer = np.array([[-2.0]])
    transition, covariance = van_loan_discretize(transfer, np.array([[3.0]]), 0.5)

    np.testing.assert_allclose(transition, [[np.exp(-1.0)]])
    np.testing.assert_allclose(covariance, [[3.0 * (1.0 - np.exp(-2.0)) / 4.0]])


def test_van_loan_nilpotent_kinematics_has_cubic_attitude_covariance():
    transfer = np.zeros((6, 6))
    transfer[3:, :3] = np.eye(3)
    velocity_psd = np.diag([2.0, 3.0, 5.0])
    noise_input = np.vstack((np.eye(3), np.zeros((3, 3))))
    dt = 0.4

    transition, covariance = van_loan_discretize(
        transfer, velocity_psd, dt, noise_input=noise_input
    )

    expected_transition = np.eye(6)
    expected_transition[3:, :3] = np.eye(3) * dt
    expected = np.block(
        [
            [velocity_psd * dt, velocity_psd * dt**2 / 2.0],
            [velocity_psd * dt**2 / 2.0, velocity_psd * dt**3 / 3.0],
        ]
    )
    np.testing.assert_allclose(transition, expected_transition, atol=1e-15)
    np.testing.assert_allclose(covariance, expected, atol=1e-15)


def test_van_loan_noise_input_and_zero_timestep():
    transfer = np.array([[0.0, 1.0], [-2.0, -0.5]])
    noise_input = np.array([[0.0], [2.0]])
    transition, covariance = van_loan_discretize(
        transfer, np.array([[0.3]]), 0.0, noise_input=noise_input
    )

    np.testing.assert_array_equal(transition, np.eye(2))
    np.testing.assert_array_equal(covariance, np.zeros((2, 2)))


def test_empty_augmented_blocks_produce_physical_psd_only():
    state = EstimatorState(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0])
    psd = assemble_continuous_process_psd(
        state, EstimatedSatellite(), unmodeled_dynamics_psd=0.05
    )

    assert psd.shape == (6, 6)
    np.testing.assert_allclose(psd, np.eye(6) * 0.05)


def test_discretize_process_noise_runs_on_wheel_and_bias_satellite():
    axis = MathConstants.unitvecs[0]
    satellite = EstimatedSatellite(
        mass=4.0,
        J_0=np.diag([3.4, 2.9, 1.3]),
        actuators=[
            RW(
                axis=axis,
                max_torque=1.0e-3,
                J=1.0e-4,
                h=0.0,
                h_max=0.1,
                bias=Bias(std_bias=2.0e-5),
                noise=Noise(),
                estimate_bias=True,
            )
        ],
        sensors=[
            Gyro(
                axis=axis,
                bias=Bias(std_bias=3.0e-5),
                noise=Noise(std_noise=1.0e-4),
                estimate_bias=True,
            )
        ],
    )
    state = EstimatorState(
        w=np.array([0.01, -0.02, 0.03]),
        q=[1.0, 0.0, 0.0, 0.0],
        h=[0.0],
        act_bias=[0.0],
        sens_bias=[0.0],
    )
    orbital_state = Orbital_State(
        ephem=Ephemeris(),
        J2000=0.22,
        R=np.array([7000.0, 0.0, 0.0]),
        V=np.array([0.0, 8.0, 0.0]),
        B=np.array([1.0e-5, 0.0, 0.0]),
        fast=True,
    )

    transition, covariance = discretize_process_noise(
        state,
        satellite,
        np.zeros(satellite.control_len),
        orbital_state,
        0.1,
        unmodeled_dynamics_psd=1.0e-8,
    )

    assert transition.shape == covariance.shape == (state.tangent_size,) * 2
    np.testing.assert_allclose(covariance, covariance.T, atol=1e-20)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-18


def test_discretize_process_noise_supports_full_and_tangent_coordinate_pairs():
    satellite = EstimatedSatellite(J_0=np.diag([0.5, 0.8, 1.2]))
    state = EstimatorState(
        w=np.array([0.02, -0.015, 0.01]),
        q=np.array([0.9, 0.2, -0.3, 0.1]) / np.linalg.norm([0.9, 0.2, -0.3, 0.1]),
    )
    orbital_state = Orbital_State(
        ephem=Ephemeris(),
        J2000=0.22,
        R=np.array([7000.0, 0.0, 0.0]),
        V=np.array([0.0, 8.0, 0.0]),
        B=np.array([1.0e-5, 0.0, 0.0]),
        fast=True,
    )
    final_state = state.copy()
    final_state.q = np.array([0.88, 0.22, -0.32, 0.12])
    final_state = final_state.normalized()
    modes = {
        "quaternion_vector": state.tangent_size,
        "full_quaternion": state.full_size,
    }

    for source_mode, source_size in modes.items():
        for target_mode, target_size in modes.items():
            transition, covariance = discretize_process_noise(
                state,
                satellite,
                np.zeros(satellite.control_len),
                orbital_state,
                0.1,
                final_state=final_state,
                source_quaternion_mode=source_mode,
                target_quaternion_mode=target_mode,
                unmodeled_dynamics_psd=1.0e-8,
            )

            assert transition.shape == (target_size, source_size)
            assert covariance.shape == (target_size, target_size)
            np.testing.assert_allclose(covariance, covariance.T, atol=1e-20)
            assert np.linalg.eigvalsh(covariance).min() >= -1e-18
