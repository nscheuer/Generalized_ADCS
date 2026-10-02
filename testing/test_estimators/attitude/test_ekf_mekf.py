"""Focused contracts for the new additive EKF and right-error MEKF."""

import numpy as np
import pytest

from ADCS.estimators.attitude_estimators import EKF, MEKF
from ADCS.estimators.attitude_estimators.attitude_estimator import AttitudeEstimator
from ADCS.estimators.process_model import propagate_state
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.errors import Noise
from ADCS.satellite_hardware.satellite import EstimatedSatellite
from ADCS.satellite_hardware.sensors import Gyro, StarTrackerQuaternion
from ADCS.state import EstimatorState, State


@pytest.fixture(scope="module")
def orbital_state() -> Orbital_State:
    return Orbital_State(
        ephem=Ephemeris(),
        J2000=0.22,
        R=np.array([7000.0, 0.0, 0.0]),
        V=np.array([0.0, 7.5, 0.0]),
        B=np.array([2.0e-5, -1.0e-5, 3.0e-5]),
        S=np.array([1.5e8, 1.0e7, -2.0e7]),
        rho=0.0,
        fast=True,
    )


def _tracker_satellite() -> EstimatedSatellite:
    tracker = StarTrackerQuaternion(noise=Noise(std_noise=np.full(4, 1.0e-3)))
    tracker.clean_reading = lambda state, orbital_state: state.q.copy()
    return EstimatedSatellite(
        J_0=np.diag([0.5, 0.8, 1.2]),
        sensors=[tracker],
    )


def _mixed_satellite() -> EstimatedSatellite:
    gyro = Gyro(axis=np.array([1.0, 0.0, 0.0]), noise=Noise(std_noise=1.0e-3))
    tracker = StarTrackerQuaternion(noise=Noise(std_noise=np.full(4, 1.0e-3)))
    tracker.clean_reading = lambda state, orbital_state: state.q.copy()
    return EstimatedSatellite(
        J_0=np.diag([0.5, 0.8, 1.2]),
        sensors=[gyro, tracker],
    )


def _estimate(*, full: bool) -> EstimatorState:
    size = 7 if full else 6
    return EstimatorState(
        w=[0.01, -0.02, 0.015],
        q=[1.0, 0.0, 0.0, 0.0],
        cov=np.eye(size) * 0.1,
        int_cov=np.zeros((size, size)),
    )


@pytest.mark.parametrize("filter_type, full", [(EKF, True), (MEKF, False)])
def test_filter_correction_reduces_attitude_error_and_keeps_covariance_valid(
    filter_type, full
):
    satellite = _tracker_satellite()
    estimator = filter_type(satellite, _estimate(full=full), dt=0.1)
    prior = estimator.state
    truth = prior.plus(
        np.array([0.0, 0.0, 0.0, 0.05, -0.025, 0.01]),
        quaternion_mode="quaternion_vector",
    )

    corrected = estimator.correct(truth.q, None)
    diagnostics = estimator.diagnostics

    assert np.linalg.norm(truth.minus(corrected)) < np.linalg.norm(truth.minus(prior))
    assert np.isclose(np.linalg.norm(corrected.q), 1.0)
    assert diagnostics["measurement_jacobian"].shape[1] == corrected.covariance.dimension
    assert diagnostics["kalman_gain"].shape == (
        corrected.covariance.dimension,
        3,
    )
    covariance = prior.cov
    jacobian = diagnostics["measurement_jacobian"]
    noise = diagnostics["measurement_noise"]
    innovation_covariance = jacobian @ covariance @ jacobian.T + noise
    expected_gain = np.linalg.solve(
        innovation_covariance, jacobian @ covariance
    ).T
    np.testing.assert_allclose(
        diagnostics["innovation_covariance"], innovation_covariance
    )
    np.testing.assert_allclose(diagnostics["kalman_gain"], expected_gain)

    identity = np.eye(covariance.shape[0])
    joseph = (
        (identity - expected_gain @ jacobian)
        @ covariance
        @ (identity - expected_gain @ jacobian).T
        + expected_gain @ noise @ expected_gain.T
    )
    reset = diagnostics["reset_jacobian"]
    np.testing.assert_allclose(corrected.cov, reset @ joseph @ reset.T, atol=1.0e-12)
    np.testing.assert_allclose(corrected.cov, corrected.cov.T, atol=1.0e-14)
    assert np.linalg.eigvalsh(corrected.cov).min() >= -1.0e-12


@pytest.mark.parametrize("filter_type, full", [(EKF, True), (MEKF, False)])
def test_prediction_has_consistent_matrix_dimensions(filter_type, full, orbital_state):
    satellite = _tracker_satellite()
    estimator = filter_type(
        satellite,
        _estimate(full=full),
        dt=0.01,
        unmodeled_dynamics_psd=1.0e-9,
    )

    predicted = estimator.predict(
        np.empty(0),
        orbital_state,
        orbital_state,
        midpoint_orbital_state=orbital_state,
    )
    diagnostics = estimator.diagnostics
    size = predicted.full_size if full else predicted.tangent_size

    assert diagnostics["transition"].shape == (size, size)
    assert diagnostics["process_noise"].shape == (size, size)
    assert predicted.covariance.shape == (size, size)
    np.testing.assert_allclose(predicted.cov, predicted.cov.T, atol=1.0e-14)
    np.testing.assert_allclose(
        diagnostics["process_noise"], diagnostics["process_noise"].T, atol=1.0e-18
    )
    assert np.isclose(np.linalg.norm(predicted.q), 1.0)
    if full:
        attitude = predicted.slice("attitude", coordinates="full")
        np.testing.assert_allclose(
            predicted.cov[attitude, attitude] @ predicted.q, 0.0, atol=1.0e-14
        )


def test_full_quaternion_transition_matches_normalized_propagation(orbital_state):
    satellite = _tracker_satellite()
    estimator = EKF(satellite, _estimate(full=True), dt=0.01)
    prior = estimator.state
    estimator.predict(
        np.empty(0),
        orbital_state,
        orbital_state,
        midpoint_orbital_state=orbital_state,
    )
    analytic = estimator.diagnostics["transition"]
    nominal = propagate_state(
        prior,
        satellite,
        np.empty(0),
        0.01,
        orbital_state,
        orbital_state,
        midpoint_orbital_state=orbital_state,
    )
    epsilon = 1.0e-7
    numerical = np.empty_like(analytic)
    for column in range(prior.full_size):
        offset = np.zeros(prior.full_size)
        offset[column] = epsilon
        plus = propagate_state(
            prior.plus(offset, quaternion_mode="full_quaternion"),
            satellite,
            np.empty(0),
            0.01,
            orbital_state,
            orbital_state,
            midpoint_orbital_state=orbital_state,
        ).minus(nominal, quaternion_mode="full_quaternion")
        minus = propagate_state(
            prior.plus(-offset, quaternion_mode="full_quaternion"),
            satellite,
            np.empty(0),
            0.01,
            orbital_state,
            orbital_state,
            midpoint_orbital_state=orbital_state,
        ).minus(nominal, quaternion_mode="full_quaternion")
        numerical[:, column] = (plus - minus) / (2.0 * epsilon)

    np.testing.assert_allclose(analytic, numerical, rtol=3.0e-6, atol=3.0e-6)


@pytest.mark.parametrize("filter_type, full", [(EKF, True), (MEKF, False)])
def test_filter_measurement_jacobian_matches_its_retraction(
    filter_type, full, orbital_state
):
    satellite = _tracker_satellite()
    estimator = filter_type(satellite, _estimate(full=full), dt=0.01)
    state = estimator.state
    stack = satellite.measurement_stack
    active = np.ones(len(stack), dtype=bool)
    reference = stack.predict(state, orbital_state, active)
    analytic = stack.jacobian(
        state,
        orbital_state,
        active,
        quaternion_mode=estimator.measurement_quaternion_mode,
        coordinates=estimator.covariance_coordinates,
    )

    epsilon = 1.0e-7
    numerical = np.empty_like(analytic)
    for column in range(analytic.shape[1]):
        offset = np.zeros(analytic.shape[1])
        offset[column] = epsilon
        plus = stack.predict(
            state.plus(offset, quaternion_mode=estimator.correction_mode),
            orbital_state,
            active,
        )
        minus = stack.predict(
            state.plus(-offset, quaternion_mode=estimator.correction_mode),
            orbital_state,
            active,
        )
        plus_error = stack.residual(
            plus,
            reference,
            active,
            quaternion_mode=estimator.measurement_quaternion_mode,
        )
        minus_error = stack.residual(
            minus,
            reference,
            active,
            quaternion_mode=estimator.measurement_quaternion_mode,
        )
        numerical[:, column] = (plus_error - minus_error) / (2.0 * epsilon)

    np.testing.assert_allclose(analytic, numerical, rtol=1.0e-7, atol=1.0e-7)


def test_ekf_projects_initial_quaternion_covariance_onto_unit_sphere():
    estimator = EKF(_tracker_satellite(), _estimate(full=True), dt=0.1)
    state = estimator.state
    attitude = state.slice("attitude", coordinates="full")

    np.testing.assert_allclose(state.cov[attitude, attitude] @ state.q, 0.0)


@pytest.mark.parametrize("filter_type, full", [(EKF, True), (MEKF, False)])
@pytest.mark.parametrize(
    "enabled, residual_size",
    [("gyro", 1), ("star_tracker_quaternion", 3), ("sensors", 4)],
)
def test_filter_accepts_individual_or_combined_sensor_selection(
    filter_type, full, enabled, residual_size
):
    satellite = _mixed_satellite()
    estimator = filter_type(satellite, _estimate(full=full), dt=0.1)
    truth = estimator.state.plus(
        np.array([0.01, 0.0, 0.0, 0.03, -0.02, 0.01]),
        quaternion_mode="quaternion_vector",
    )
    measurements = satellite.measurement_stack.predict(truth, None)

    estimator.correct(measurements, None, enabled=enabled)

    diagnostics = estimator.diagnostics
    assert diagnostics["innovation"].shape == (residual_size,)
    assert diagnostics["measurement_jacobian"].shape == (
        residual_size,
        estimator.state.covariance.dimension,
    )


@pytest.mark.parametrize("filter_type, full", [(EKF, True), (MEKF, False)])
def test_no_active_measurement_is_a_noop(filter_type, full):
    satellite = _tracker_satellite()
    estimator = filter_type(satellite, _estimate(full=full), dt=0.1)
    before = estimator.state

    after = estimator.correct(np.full(4, np.nan), None)

    assert after == before
    assert estimator.diagnostics["innovation"].size == 0
    assert estimator.diagnostics["kalman_gain"].shape == (
        before.covariance.dimension,
        0,
    )


def test_filters_reject_the_other_attitude_covariance_layout():
    satellite = _tracker_satellite()
    with pytest.raises(ValueError, match="EKF requires a 7x7 full covariance"):
        EKF(satellite, _estimate(full=False), dt=0.1)
    with pytest.raises(ValueError, match="MEKF requires a 6x6 tangent covariance"):
        MEKF(satellite, _estimate(full=True), dt=0.1)


def test_first_generation_filters_explicitly_reject_augmented_states():
    satellite = EstimatedSatellite(
        sensors=[Gyro(axis=np.array([1.0, 0.0, 0.0]), estimate_bias=True)]
    )
    state = EstimatorState(
        w=np.zeros(3),
        q=[1.0, 0.0, 0.0, 0.0],
        sens_bias=[0.0],
        cov=np.eye(7),
        int_cov=np.zeros((7, 7)),
    )

    with pytest.raises(NotImplementedError, match="does not yet support estimated biases"):
        MEKF(satellite, state, dt=0.1)


# --- chart configuration ---------------------------------------------------


def _identity_state(size: int = 6) -> EstimatorState:
    return EstimatorState(
        w=np.zeros(3),
        q=[1.0, 0.0, 0.0, 0.0],
        cov=np.eye(size) * 1.0e-2,
        int_cov=np.zeros((size, size)),
    )


@pytest.mark.parametrize(
    "covariance_coordinates, correction_mode, measurement_quaternion_mode",
    [
        ("tangent", "mrp", "quaternion_vector"),  # H in one chart, P in another
        ("tangent", "full_quaternion", "quaternion_vector"),
        ("full", "mrp", "mrp"),
        ("full", "full_quaternion", "full_quaternion"),  # residuals are always minimal
        ("tangent", "bogus", "bogus"),
    ],
)
def test_attitude_estimator_rejects_inconsistent_charts_at_construction(
    covariance_coordinates, correction_mode, measurement_quaternion_mode
):
    size = 7 if covariance_coordinates == "full" else 6
    with pytest.raises(ValueError):
        AttitudeEstimator(
            _tracker_satellite(),
            _identity_state(size),
            dt=1.0,
            covariance_coordinates=covariance_coordinates,
            correction_mode=correction_mode,
            measurement_quaternion_mode=measurement_quaternion_mode,
        )


@pytest.mark.parametrize("mode", ["rotation_vector", "mrp", "two_mrp", "cayley"])
def test_mekf_accepts_every_three_parameter_chart(mode):
    estimator = MEKF(_tracker_satellite(), _identity_state(), dt=1.0, quaternion_mode=mode)
    assert estimator.correction_mode == mode
    assert estimator.measurement_quaternion_mode == mode


def test_mekf_rejects_full_quaternion_chart_at_construction():
    with pytest.raises(ValueError):
        MEKF(_tracker_satellite(), _identity_state(), dt=1.0, quaternion_mode="full_quaternion")


def test_chart_configuration_is_read_only():
    estimator = MEKF(_tracker_satellite(), _identity_state(), dt=1.0)
    for name in ("covariance_coordinates", "correction_mode", "measurement_quaternion_mode"):
        with pytest.raises(AttributeError):
            setattr(estimator, name, "mrp")
