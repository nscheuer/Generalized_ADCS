"""Tests that kill the mutants surviving testing/test_estimators/test_attitude_ukf.py (proposed additions)."""
from __future__ import annotations

import numpy as np
import pytest

from ADCS.estimators.attitude_estimators import UKF
from ADCS.helpers.math_helpers import quat_diff, quat_mult
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.errors import Noise
from ADCS.satellite_hardware.satellite import EstimatedSatellite
from ADCS.satellite_hardware.sensors import StarTrackerQuaternion
from ADCS.state import EstimatorState, State


@pytest.fixture()
def orbital_state() -> Orbital_State:
    return Orbital_State(
        ephem=Ephemeris(), J2000=0.22, R=np.array([7000.0, 0.0, 0.0]), V=np.array([0.0, 7.5, 0.0]), fast=True
    )


def _satellite() -> EstimatedSatellite:
    tracker = StarTrackerQuaternion(noise=Noise(std_noise=np.full(4, 1.0e-3)))

    def clean(*args, **kwargs):
        state = args[0] if args else kwargs["x"]
        return state.q.copy()

    tracker.clean_reading = clean
    return EstimatedSatellite(J_0=np.diag([0.5, 0.8, 1.2]), sensors=[tracker])


def _state(attitude_variance: float = 0.1) -> EstimatorState:
    return EstimatorState(
        w=[0.01, -0.02, 0.015], q=[1.0, 0.0, 0.0, 0.0],
        cov=np.diag([0.1] * 3 + [attitude_variance] * 3), int_cov=np.zeros((6, 6)),
    )


def _reconstruct(diagnostics: dict) -> np.ndarray:
    deviations = diagnostics["predicted_sigma_deviations"]
    return np.einsum("i,ij,ik->jk", diagnostics["sigma_weights_covariance"], deviations, deviations)


def test_sigma_weights_and_offsets_match_the_scaled_unscented_closed_form(orbital_state):
    """Kills M01 (weight sign, equivalent at alpha=1), M11 (W0c sign), M12 (gamma = scale)."""
    alpha, beta, kappa, n = 0.5, 2.0, 1.0, 6
    estimator = UKF(_satellite(), _state(), dt=0.1, alpha=alpha, beta=beta, kappa=kappa)
    prior = estimator.state
    estimator.predict(np.empty(0), orbital_state, orbital_state)
    diagnostics = estimator.diagnostics

    lam = alpha**2 * (n + kappa) - n
    mean_weights = np.full(2 * n + 1, 0.5 / (n + lam))
    mean_weights[0] = lam / (n + lam)
    covariance_weights = mean_weights.copy()
    covariance_weights[0] += 1.0 - alpha**2 + beta
    np.testing.assert_allclose(diagnostics["sigma_weights_mean"], mean_weights, atol=1e-14)
    np.testing.assert_allclose(diagnostics["sigma_weights_covariance"], covariance_weights, atol=1e-14)
    assert diagnostics["sigma_weights_mean"].sum() == pytest.approx(1.0, abs=1e-13)

    gamma = np.sqrt(n + lam)
    upper = np.linalg.cholesky(prior.cov).T
    np.testing.assert_allclose(diagnostics["sigma_offsets"][:n], gamma * upper, atol=1e-12)
    np.testing.assert_allclose(diagnostics["sigma_offsets"][n:], -gamma * upper, atol=1e-12)


@pytest.mark.parametrize("mode", ["quaternion_vector", "rotation_vector", "mrp"])
def test_zero_length_prediction_reproduces_the_prior_covariance(orbital_state, mode):
    """Kills M06 (wrong chart in the prediction deviations) and M12 (wrong sigma scale)."""
    estimator = UKF(_satellite(), _state(0.1), dt=0.1, quaternion_mode=mode, unmodeled_dynamics_psd=0.0)
    prior = estimator.state
    predicted = estimator.predict(np.empty(0), orbital_state, orbital_state, dt=0.0)
    np.testing.assert_allclose(_reconstruct(estimator.diagnostics), prior.cov, atol=1e-12)
    np.testing.assert_allclose(predicted.cov, prior.cov, atol=1e-12)


def test_prediction_adds_the_discretised_process_noise_exactly_once(orbital_state):
    """Kills M02 (process noise dropped) and M15 (process noise doubled)."""
    estimator = UKF(_satellite(), _state(), dt=0.1, unmodeled_dynamics_psd=1.0e-6)
    predicted = estimator.predict(np.empty(0), orbital_state, orbital_state)
    diagnostics = estimator.diagnostics
    assert np.linalg.norm(diagnostics["process_noise"]) > 0.0
    np.testing.assert_allclose(
        predicted.cov - _reconstruct(diagnostics), diagnostics["process_noise"], atol=1e-15, rtol=1e-9
    )


def test_predicted_state_is_the_weighted_manifold_mean_of_the_propagated_points(orbital_state):
    """Kills M05 (manifold-mean iteration skipped: the propagated centre point is returned)."""
    estimator = UKF(_satellite(), _state(0.5), dt=0.1)
    estimator.predict(np.empty(0), orbital_state, orbital_state)
    diagnostics = estimator.diagnostics
    residual = diagnostics["sigma_weights_mean"] @ diagnostics["predicted_sigma_deviations"]
    assert np.linalg.norm(residual) < 1.0e-10


def test_quaternion_vector_sigma_points_are_clamped_inside_the_chart(orbital_state):
    """Kills M10 (clamp removed); the shipped test writes to a covariance copy and never activates the clamp."""
    estimator = UKF(_satellite(), _state(1.0), dt=0.1)
    estimator.predict(np.empty(0), orbital_state, orbital_state)
    attitude_offsets = estimator.diagnostics["sigma_offsets"][:, 3:6]
    largest = np.linalg.norm(attitude_offsets, axis=1).max()
    assert largest == pytest.approx(1.9, abs=1.0e-12)
    assert largest < np.sqrt(6.0)  # the unclamped spread would be sqrt(n + lambda) = 2.449


def test_correction_satisfies_the_unscented_gain_and_reset_identities(orbital_state):
    """Kills M08 (prior covariance kept), M14 (reset transport skipped), M16 (R dropped from the gain),
    M18 (correction not K @ innovation)."""
    estimator = UKF(_satellite(), _state(), dt=0.1)
    prior = estimator.state
    truth = prior.plus([0.0, 0.0, 0.0, 0.05, -0.025, 0.01])
    corrected = estimator.correct(truth.q, orbital_state)
    d = estimator.diagnostics
    gain, innovation_covariance, innovation, reset = (
        d["kalman_gain"], d["innovation_covariance"], d["innovation"], d["reset_jacobian"]
    )
    state_deviations = np.vstack((np.zeros(6), d["sigma_offsets"]))
    cross = np.einsum(
        "i,ij,ik->jk", d["sigma_weights_covariance"], state_deviations, d["measurement_sigma_deviations"]
    )
    np.testing.assert_allclose(gain @ innovation_covariance, cross, atol=1e-12)
    np.testing.assert_allclose(d["correction"], gain @ innovation, atol=1e-14)
    posterior = prior.cov - gain @ innovation_covariance @ gain.T
    np.testing.assert_allclose(corrected.cov, reset @ posterior @ reset.T, atol=1e-12)
    assert np.trace(corrected.cov) < np.trace(prior.cov)
    assert not np.allclose(reset, np.eye(6), atol=1e-6)


def test_measurement_mean_is_the_weighted_manifold_mean_of_asymmetric_quaternions(orbital_state):
    """Kills M13 (measurement-mean iteration skipped). Symmetric star-tracker sigma points never exercise
    the iteration, so asymmetric synthetic sigma measurements are needed."""
    estimator = UKF(_satellite(), _state(), dt=0.1)
    stack = estimator.satellite.measurement_stack
    q0 = estimator.state.q
    offsets = [np.zeros(3), [0.6, 0.0, 0.0], [-0.2, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, -0.1, 0.0]]
    values = [quat_mult(q0, State.quaternion_delta_from_vector(offset)) for offset in offsets]
    weights = np.full(len(values), 0.2)
    mean = estimator._measurement_mean(stack, values, np.array([True]), weights)
    deviations = np.vstack([State.quaternion_delta_to_vector(quat_diff(mean, value)) for value in values])
    assert np.linalg.norm(weights @ deviations) < 1.0e-10
    assert np.linalg.norm(State.quaternion_delta_to_vector(quat_diff(values[0], mean))) > 0.1
