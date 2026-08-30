"""Fast integration contracts for the next estimator implementations."""

from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from ADCS.covariance import Covariance
from ADCS.estimators.process_noise import discretize_process_noise
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.errors import Noise
from ADCS.satellite_hardware.satellite import EstimatedSatellite
from ADCS.satellite_hardware.sensors import (
    EarthHorizonSensor,
    Gyro,
    MTM,
    StarTracker,
    StarTrackerQuaternion,
    SunPair,
    SunSensor,
)
from ADCS.state import EstimatorState


@pytest.fixture(scope="module")
def orbital_state() -> Orbital_State:
    result = Orbital_State(
        ephem=Ephemeris(),
        J2000=0.22,
        R=np.array([7000.0, 0.0, 0.0]),
        V=np.array([0.0, 7.5, 0.0]),
        B=np.array([2.0e-5, -1.0e-5, 3.0e-5]),
        S=np.array([1.5e8, 1.0e7, -2.0e7]),
        rho=0.0,
        fast=True,
    )
    result._sunlit = True
    return result


def _estimate(*, wheel_momentum=()) -> EstimatorState:
    quaternion = np.array([0.9, 0.2, -0.3, 0.1])
    quaternion /= np.linalg.norm(quaternion)
    return EstimatorState(
        w=np.array([0.02, -0.015, 0.01]),
        q=quaternion,
        h=wheel_momentum,
    )


def _as_estimate(state) -> EstimatorState:
    return EstimatorState(w=state.w, q=state.q, h=state.h)


def _numerical_discrete_transition(
    satellite: EstimatedSatellite,
    state: EstimatorState,
    orbital_state: Orbital_State,
    dt: float,
) -> np.ndarray:
    control = np.zeros(satellite.control_len)

    def propagate(candidate: EstimatorState) -> EstimatorState:
        return _as_estimate(
            satellite.noiseless_rk4(
                candidate,
                control,
                dt,
                orbital_state,
                orbital_state,
                mid_orbital_state=orbital_state,
                quat_as_vec=True,
            )
        )

    nominal = propagate(state)
    epsilon = 1.0e-7
    transition = np.empty((state.tangent_size, state.tangent_size))
    for column in range(state.tangent_size):
        offset = np.zeros(state.tangent_size)
        offset[column] = epsilon
        plus = propagate(state.plus(offset)).minus(nominal)
        minus = propagate(state.plus(-offset)).minus(nominal)
        transition[:, column] = (plus - minus) / (2.0 * epsilon)
    return transition


@pytest.mark.parametrize(
    "wheel_count", [0, 1, 3], ids=["no_wheel", "one_wheel", "three_wheels"]
)
def test_discrete_transition_matches_nonlinear_propagation(
    orbital_state, wheel_count
):
    if wheel_count:
        actuators = [
            RW(
                axis=np.eye(3)[index],
                max_torque=0.01,
                J=0.001,
                h=0.02,
                h_max=0.1,
            )
            for index in range(wheel_count)
        ]
        state = _estimate(wheel_momentum=np.full(wheel_count, 0.02))
    else:
        actuators = [MTQ(axis=np.array([1.0, 0.0, 0.0]), max_torque=1.0)]
        state = _estimate()
    satellite = EstimatedSatellite(
        J_0=np.diag([0.5, 0.8, 1.2]),
        actuators=actuators,
    )
    dt = 0.01

    analytic, _ = discretize_process_noise(
        state,
        satellite,
        np.zeros(satellite.control_len),
        orbital_state,
        dt,
        unmodeled_dynamics_psd=0.0,
    )
    numerical = _numerical_discrete_transition(satellite, state, orbital_state, dt)

    np.testing.assert_allclose(analytic, numerical, rtol=2.0e-6, atol=2.0e-6)


def test_builtin_measurement_stack_jacobian_matches_local_finite_difference(
    orbital_state,
):
    star = SimpleNamespace(
        s_eci=np.array([0.3, -0.4, np.sqrt(0.75)]),
        vmag=1.0,
    )
    vector_tracker = StarTracker(fov=np.pi)
    vector_tracker._select_star = lambda q, os: star
    quaternion_tracker = StarTrackerQuaternion(fov=np.pi, min_stars=1)
    quaternion_tracker._select_stars = lambda q, os: [star]
    sensors = [
        Gyro(axis=np.array([1.0, 2.0, -1.0])),
        MTM(axis=np.array([-1.0, 2.0, 1.0])),
        SunSensor(axis=np.array([1.0, 0.2, -0.1]), efficiency=0.8),
        SunPair(axis=np.array([0.2, 1.0, 0.3]), efficiency=(0.7, 0.9)),
        vector_tracker,
        quaternion_tracker,
        EarthHorizonSensor(boresight=np.array([-1.0, 0.0, 0.0]), fov=np.pi),
    ]
    stack = EstimatedSatellite(sensors=sensors).measurement_stack
    state = _estimate()
    active = np.ones(len(stack.entries), dtype=bool)
    predicted = stack.predict(state, orbital_state, active)
    analytic = stack.jacobian(state, orbital_state, active)

    epsilon = 1.0e-7
    numerical = np.empty_like(analytic)
    for column in range(state.tangent_size):
        offset = np.zeros(state.tangent_size)
        offset[column] = epsilon
        plus = stack.predict(state.plus(offset), orbital_state, active)
        minus = stack.predict(state.plus(-offset), orbital_state, active)
        plus_local = stack.residual(plus, predicted, active)
        minus_local = stack.residual(minus, predicted, active)
        numerical[:, column] = (plus_local - minus_local) / (2.0 * epsilon)

    np.testing.assert_allclose(analytic, numerical, rtol=2.0e-5, atol=2.0e-8)


def test_quaternion_innovation_and_correction_have_the_same_sign():
    tracker = StarTrackerQuaternion(noise=Noise(std_noise=np.full(4, 1.0e-3)))
    stack = EstimatedSatellite(sensors=[tracker]).measurement_stack
    predicted = _estimate()
    true_delta = np.zeros(predicted.tangent_size)
    true_delta[predicted.tangent_slices["attitude"]] = [0.04, -0.02, 0.01]
    truth = predicted.plus(true_delta)
    active = np.array([True])

    innovation = stack.residual(truth.q, predicted.q, active)
    H = stack.jacobian(predicted, None, active)
    R = stack.covariance(predicted, active)
    prior = Covariance.identity(predicted.tangent_size, scale=0.1)
    gain, posterior = prior.updated_linear(H, R)
    corrected = predicted.plus(gain @ innovation)

    np.testing.assert_allclose(innovation, true_delta[3:6], atol=1.0e-14)
    assert np.linalg.norm(truth.minus(corrected)) < np.linalg.norm(
        truth.minus(predicted)
    )
    assert np.isclose(np.linalg.norm(corrected.q), 1.0)
    np.testing.assert_allclose(posterior.as_matrix(), posterior.as_matrix().T)
    assert np.linalg.eigvalsh(posterior.as_matrix()).min() >= -1.0e-14


def test_every_measurement_subset_has_consistent_compact_dimensions():
    tracker = StarTrackerQuaternion(noise=Noise(std_noise=np.full(4, 0.1)))
    tracker.clean_reading = lambda state, os: state.q.copy()
    wheel = RW(
        axis=np.array([1.0, 0.0, 0.0]),
        max_torque=0.1,
        J=0.01,
        h=0.2,
        h_max=1.0,
        h_meas_noise=Noise(std_noise=0.2),
    )
    satellite = EstimatedSatellite(
        sensors=[
            Gyro(
                axis=np.array([1.0, 0.0, 0.0]),
                noise=Noise(std_noise=0.1),
            ),
            tracker,
        ],
        actuators=[wheel],
    )
    stack = satellite.measurement_stack
    state = _estimate(wheel_momentum=[0.2])
    measured = stack.predict(state, None)

    residual_sizes = [1, 3, 1]
    for selected in product((False, True), repeat=len(stack.entries)):
        active = np.asarray(selected)
        predicted = stack.predict(state, None, active)
        expected_size = sum(size for size, use in zip(residual_sizes, active) if use)

        assert stack.residual(measured, predicted, active).shape == (expected_size,)
        assert stack.jacobian(state, None, active).shape == (
            expected_size,
            state.tangent_size,
        )
        assert stack.covariance(state, active).shape == (expected_size, expected_size)


def test_linear_ekf_primitives_match_in_full_and_square_root_forms():
    initial = np.array([[2.0, 0.3], [0.3, 1.0]])
    transition = np.array([[1.0, 0.2], [0.0, 1.0]])
    process_noise = np.diag([0.1, 0.2])
    measurement_jacobian = np.array([[1.0, -0.5]])
    measurement_noise = np.array([[0.3]])
    expected_prediction = transition @ initial @ transition.T + process_noise
    innovation_covariance = (
        measurement_jacobian @ expected_prediction @ measurement_jacobian.T
        + measurement_noise
    )
    expected_gain = (
        expected_prediction
        @ measurement_jacobian.T
        @ np.linalg.inv(innovation_covariance)
    )
    residual = np.eye(2) - expected_gain @ measurement_jacobian
    expected_posterior = (
        residual @ expected_prediction @ residual.T
        + expected_gain @ measurement_noise @ expected_gain.T
    )

    results = []
    for form in ("full", "sqrt"):
        predicted = Covariance(initial, form=form).predicted_linear(
            transition, process_noise
        )
        gain, posterior = predicted.updated_linear(
            measurement_jacobian, measurement_noise
        )
        results.append((gain, posterior.as_matrix()))

    np.testing.assert_allclose(results[0][0], results[1][0])
    np.testing.assert_allclose(results[0][1], results[1][1])
    np.testing.assert_allclose(results[0][0], expected_gain)
    np.testing.assert_allclose(results[0][1], expected_posterior)
    np.testing.assert_allclose(results[0][1], results[0][1].T)
    assert np.linalg.eigvalsh(results[0][1]).min() >= -1.0e-14
