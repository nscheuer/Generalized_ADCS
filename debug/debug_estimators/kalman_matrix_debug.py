"""Small deterministic matrix audit shared by the EKF and MEKF debug scripts."""

from __future__ import annotations

import numpy as np

from ADCS.estimators.attitude_estimators import AttitudeEstimator, EKF, MEKF
from ADCS.estimators.process_model import propagate_state
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.errors import Noise
from ADCS.satellite_hardware.satellite import EstimatedSatellite
from ADCS.satellite_hardware.sensors import Gyro, StarTrackerQuaternion
from ADCS.state import EstimatorState


def _satellite() -> EstimatedSatellite:
    gyros = [
        Gyro(axis=axis, noise=Noise(std_noise=2.0e-4))
        for axis in np.eye(3)
    ]
    tracker = StarTrackerQuaternion(noise=Noise(std_noise=np.full(4, 1.0e-3)))

    # This audit targets estimator plumbing, not star visibility. A perfect
    # quaternion observation keeps the example deterministic and inexpensive.
    tracker.clean_reading = lambda state, orbital_state: state.q.copy()
    return EstimatedSatellite(
        J_0=np.diag([0.5, 0.8, 1.2]),
        sensors=[*gyros, tracker],
    )


def _orbital_state() -> Orbital_State:
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


def _initial_estimate(full_quaternion: bool) -> EstimatorState:
    dimension = 7 if full_quaternion else 6
    diagonal = np.array(
        [4.0e-4, 4.0e-4, 4.0e-4]
        + ([0.05] * 4 if full_quaternion else [0.05] * 3)
    )
    return EstimatorState(
        w=np.array([0.01, -0.02, 0.015]),
        q=np.array([0.96, 0.12, -0.18, 0.16]),
        cov=np.diag(diagonal),
        int_cov=np.zeros((dimension, dimension)),
    )


def _numerical_transition(
    estimator: AttitudeEstimator,
    prior: EstimatorState,
    nominal: EstimatorState,
    orbital_state: Orbital_State,
    dt: float,
) -> np.ndarray:
    dimension = prior.covariance.dimension
    numerical = np.empty((dimension, dimension))
    epsilon = 1.0e-7
    control = np.empty(0)
    for column in range(dimension):
        offset = np.zeros(dimension)
        offset[column] = epsilon
        plus = propagate_state(
            prior.plus(offset, quaternion_mode=estimator.correction_mode),
            estimator.satellite,
            control,
            dt,
            orbital_state,
            orbital_state,
            midpoint_orbital_state=orbital_state,
        ).minus(nominal, quaternion_mode=estimator.correction_mode)
        minus = propagate_state(
            prior.plus(-offset, quaternion_mode=estimator.correction_mode),
            estimator.satellite,
            control,
            dt,
            orbital_state,
            orbital_state,
            midpoint_orbital_state=orbital_state,
        ).minus(nominal, quaternion_mode=estimator.correction_mode)
        numerical[:, column] = (plus - minus) / (2.0 * epsilon)
    return numerical


def _numerical_measurement_jacobian(
    estimator: AttitudeEstimator,
    state: EstimatorState,
    orbital_state: Orbital_State,
    active: np.ndarray,
) -> np.ndarray:
    stack = estimator.satellite.measurement_stack
    reference = stack.predict(state, orbital_state, active)
    dimension = state.covariance.dimension
    residual_size = stack.residual(
        reference,
        reference,
        active,
        quaternion_mode=estimator.measurement_quaternion_mode,
    ).size
    numerical = np.empty((residual_size, dimension))
    epsilon = 1.0e-7
    for column in range(dimension):
        offset = np.zeros(dimension)
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
    return numerical


def _print_matrix(name: str, matrix: np.ndarray) -> None:
    print(f"\n{name}  shape={matrix.shape}")
    print(matrix)


def run_matrix_debug(filter_type: type[EKF] | type[MEKF]) -> None:
    """Run one prediction/update and audit every important EKF matrix."""
    np.set_printoptions(precision=6, suppress=False, linewidth=180)
    full_quaternion = filter_type is EKF
    dt = 0.01
    satellite = _satellite()
    orbital_state = _orbital_state()
    estimator = filter_type(
        satellite,
        _initial_estimate(full_quaternion),
        dt=dt,
        unmodeled_dynamics_psd=1.0e-9,
    )

    initial_estimate = estimator.state
    truth = initial_estimate.plus(
        np.array([2.0e-3, -1.0e-3, 1.5e-3, 0.04, -0.025, 0.015]),
        quaternion_mode="quaternion_vector",
    )
    truth = propagate_state(
        truth,
        satellite,
        np.empty(0),
        dt,
        orbital_state,
        orbital_state,
        midpoint_orbital_state=orbital_state,
    )

    predicted = estimator.predict(
        np.empty(0),
        orbital_state,
        orbital_state,
        midpoint_orbital_state=orbital_state,
    )
    prediction_diagnostics = estimator.diagnostics
    numerical_transition = _numerical_transition(
        estimator, initial_estimate, predicted, orbital_state, dt
    )

    stack = satellite.measurement_stack
    active = np.ones(len(stack), dtype=bool)
    analytical_jacobian = stack.jacobian(
        predicted,
        orbital_state,
        active,
        quaternion_mode=estimator.measurement_quaternion_mode,
        coordinates=estimator.covariance_coordinates,
    )
    numerical_jacobian = _numerical_measurement_jacobian(
        estimator, predicted, orbital_state, active
    )
    measurements = stack.predict(truth, orbital_state, active)
    attitude_error_before = np.linalg.norm(truth.minus(predicted)[3:6])
    prior_covariance = predicted.cov.copy()

    corrected = estimator.correct(measurements, orbital_state)
    diagnostics = estimator.diagnostics
    attitude_error_after = np.linalg.norm(truth.minus(corrected)[3:6])

    transition = prediction_diagnostics["transition"]
    process_noise = prediction_diagnostics["process_noise"]
    jacobian = diagnostics["measurement_jacobian"]
    measurement_noise = diagnostics["measurement_noise"]
    innovation_covariance = diagnostics["innovation_covariance"]
    gain = diagnostics["kalman_gain"]
    reset = diagnostics["reset_jacobian"]

    expected_innovation_covariance = (
        jacobian @ prior_covariance @ jacobian.T + measurement_noise
    )
    expected_gain = np.linalg.solve(
        expected_innovation_covariance, jacobian @ prior_covariance
    ).T
    identity = np.eye(prior_covariance.shape[0])
    residual_update = identity - gain @ jacobian
    joseph_covariance = (
        residual_update @ prior_covariance @ residual_update.T
        + gain @ measurement_noise @ gain.T
    )
    expected_covariance = reset @ joseph_covariance @ reset.T

    matrices = {
        "Phi (state transition)": transition,
        "Qd (discrete process noise)": process_noise,
        "H (measurement Jacobian)": jacobian,
        "R (measurement noise)": measurement_noise,
        "S (innovation covariance)": innovation_covariance,
        "K (Kalman gain)": gain,
        "J_reset (covariance transport)": reset,
        "P+ (corrected covariance)": corrected.cov,
    }
    print(f"\n{filter_type.__name__} matrix audit")
    print(f"coordinates: {estimator.covariance_coordinates}")
    print(f"measurement sources: {stack.names}")
    for name, matrix in matrices.items():
        _print_matrix(name, matrix)

    errors = {
        "Phi vs finite difference": np.max(np.abs(transition - numerical_transition)),
        "H vs finite difference": np.max(np.abs(analytical_jacobian - numerical_jacobian)),
        "S algebra": np.max(
            np.abs(innovation_covariance - expected_innovation_covariance)
        ),
        "K algebra": np.max(np.abs(gain - expected_gain)),
        "Joseph + reset covariance": np.max(
            np.abs(corrected.cov - expected_covariance)
        ),
        "P symmetry": np.max(np.abs(corrected.cov - corrected.cov.T)),
        "Qd symmetry": np.max(np.abs(process_noise - process_noise.T)),
    }
    print("\nNumerical checks")
    for name, error in errors.items():
        print(f"  {name:<30} max error = {error:.3e}")
    print(f"  minimum eigenvalue(P+)       = {np.linalg.eigvalsh(corrected.cov).min():.3e}")
    print(
        "  attitude error before/after  = "
        f"{attitude_error_before:.3e} / {attitude_error_after:.3e}"
    )
    print(f"  quaternion norm              = {np.linalg.norm(corrected.q):.16f}")

    assert errors["Phi vs finite difference"] < 4.0e-6
    assert errors["H vs finite difference"] < 2.0e-7
    assert errors["S algebra"] < 1.0e-12
    assert errors["K algebra"] < 1.0e-10
    assert errors["Joseph + reset covariance"] < 1.0e-12
    assert errors["P symmetry"] < 1.0e-12
    assert errors["Qd symmetry"] < 1.0e-15
    assert np.linalg.eigvalsh(corrected.cov).min() >= -1.0e-12
    assert attitude_error_after < attitude_error_before
    assert np.isclose(np.linalg.norm(corrected.q), 1.0)
    print("\nPASS")
