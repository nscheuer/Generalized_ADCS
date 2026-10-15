"""Contracts for the non-augmented tangent-space attitude UKF."""

from __future__ import annotations

import numpy as np
import pytest

from ADCS.estimators.attitude_estimators import SRUKF, UKF
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.errors import Noise
from ADCS.satellite_hardware.actuators import MTQ
from ADCS.satellite_hardware.satellite import EstimatedSatellite
from ADCS.satellite_hardware.sensors import Gyro, StarTrackerQuaternion
from ADCS.state import EstimatorState


@pytest.fixture()
def orbital_state() -> Orbital_State:
    return Orbital_State(
        ephem=Ephemeris(),
        J2000=0.22,
        R=np.array([7000.0, 0.0, 0.0]),
        V=np.array([0.0, 7.5, 0.0]),
        fast=True,
    )


def _satellite() -> EstimatedSatellite:
    tracker = StarTrackerQuaternion(noise=Noise(std_noise=np.full(4, 1.0e-3)))
    tracker.clean_reading = lambda state, orbital_state: state.q.copy()
    return EstimatedSatellite(J_0=np.diag([0.5, 0.8, 1.2]), sensors=[tracker])


def _state() -> EstimatorState:
    return EstimatorState(
        w=[0.01, -0.02, 0.015],
        q=[1.0, 0.0, 0.0, 0.0],
        cov=np.eye(6) * 0.1,
        int_cov=np.zeros((6, 6)),
    )


def _actuated_satellite(control_std: list[float]) -> EstimatedSatellite:
    return EstimatedSatellite(
        J_0=np.diag([0.5, 0.8, 1.2]),
        actuators=[
            MTQ(
                axis=np.eye(3)[index],
                max_torque=1.0,
                noise=Noise(std_noise=control_std[index]),
            )
            for index in range(3)
        ],
    )


def test_ukf_uses_only_state_covariance_sigma_points(orbital_state):
    estimator = UKF(
        _satellite(), _state(), dt=0.1, unmodeled_dynamics_psd=1.0e-9
    )

    predicted = estimator.predict(np.empty(0), orbital_state, orbital_state)
    diagnostics = estimator.diagnostics

    assert diagnostics["sigma_offsets"].shape == (12, 6)
    assert diagnostics["sigma_weights_mean"].shape == (13,)
    assert diagnostics["predicted_sigma_deviations"].shape == (13, 6)
    assert predicted.covariance.shape == (6, 6)
    assert np.linalg.eigvalsh(predicted.cov).min() >= -1.0e-12


def test_ukf_limits_quaternion_vector_sigma_points_to_the_chart_domain(
    orbital_state,
):
    state = _state()
    covariance = state.cov
    covariance[3:6, 3:6] = np.eye(3)
    state.cov = covariance
    estimator = UKF(_satellite(), state, dt=0.1)

    estimator.predict(np.empty(0), orbital_state, orbital_state)

    attitude_offsets = estimator.diagnostics["sigma_offsets"][:, 3:6]
    assert np.linalg.norm(attitude_offsets, axis=1).max() <= 1.9 + 1.0e-12


@pytest.mark.parametrize("filter_type", [UKF, SRUKF])
@pytest.mark.parametrize("variance", [1.0, 2.0, 4.0, 9.0])
def test_large_quaternion_vector_prior_predicts_without_losing_covariance(
    filter_type, variance, orbital_state
):
    state = _state()
    state.cov = np.diag([1.0e-6] * 3 + [variance] * 3)
    estimator = filter_type(_satellite(), state, dt=1.0)
    predicted = estimator.predict(np.empty(0), orbital_state, orbital_state)
    diagnostics = estimator.diagnostics

    assert np.isclose(np.linalg.norm(predicted.q), 1.0)
    assert np.linalg.norm(diagnostics["sigma_offsets"][:, 3:6], axis=1).max() <= 1.9 + 1e-12
    np.testing.assert_allclose(
        diagnostics["sigma_weights_mean"] @ diagnostics["predicted_sigma_deviations"],
        0.0, atol=2e-12,
    )
    assert np.linalg.eigvalsh(predicted.cov).min() >= -1e-12
    # Free rigid-body propagation preserves the initial attitude spread; the
    # tiny rate uncertainty adds only a small positive contribution.
    np.testing.assert_allclose(np.trace(predicted.cov[3:6, 3:6]), 3 * variance, rtol=1e-5)
    truth = predicted.plus([0, 0, 0, 0.25, -0.1, 0.05], quaternion_mode="rotation_vector")
    corrected = estimator.correct(truth.q, orbital_state)
    assert np.linalg.norm(truth.minus(corrected, quaternion_mode="rotation_vector")) < (
        np.linalg.norm(truth.minus(predicted, quaternion_mode="rotation_vector"))
    )
    assert np.isclose(np.linalg.norm(corrected.q), 1.0)
    assert np.trace(corrected.cov[3:6, 3:6]) < np.trace(predicted.cov[3:6, 3:6])
    assert np.linalg.eigvalsh(corrected.cov).min() >= -1e-12
    assert corrected.covariance.form == ("sqrt" if filter_type is SRUKF else "full")


@pytest.mark.parametrize("filter_type", [UKF, SRUKF])
def test_cayley_prediction_has_zero_weighted_residual(filter_type, orbital_state):
    state = _state()
    state.cov = np.diag([0.1] * 3 + [1.0] * 3)
    estimator = filter_type(_satellite(), state, dt=1.0, quaternion_mode="cayley")
    predicted = estimator.predict(np.empty(0), orbital_state, orbital_state)
    diagnostics = estimator.diagnostics
    np.testing.assert_allclose(
        diagnostics["sigma_weights_mean"] @ diagnostics["predicted_sigma_deviations"],
        0.0, atol=2e-12,
    )
    assert np.isclose(np.linalg.norm(predicted.q), 1.0)
    assert np.linalg.eigvalsh(predicted.cov).min() >= -1e-12


@pytest.mark.parametrize("filter_type", [UKF, SRUKF])
def test_state_and_measurement_means_allow_signed_weights_outside_retraction_domain(filter_type):
    estimator = filter_type(_satellite(), _state(), dt=1.0)
    weights = np.array([-9.0, 5.0, 5.0])
    points = [
        _state().plus([0, 0, 0, angle, 0, 0], quaternion_mode="rotation_vector")
        for angle in [0.0, 1.0, 0.0]
    ]
    values = np.vstack([point.q for point in points])
    # The old mean's first correction exceeded the quaternion-vector norm 2.
    assert np.linalg.norm(weights @ (2.0 * values[:, 1:])) > 2.0
    expected = weights @ values
    expected /= np.linalg.norm(expected)
    state_mean = estimator._state_mean(points, weights)
    measurement_mean = estimator._measurement_mean(
        estimator.satellite.measurement_stack, [point.q for point in points],
        np.array([True]), weights,
    )
    np.testing.assert_allclose(state_mean.q, expected, atol=1e-12)
    np.testing.assert_allclose(measurement_mean, expected, atol=1e-12)


def test_clamped_sigma_weights_use_effective_alpha():
    state = _state()
    state.cov = np.diag([1e-6] * 3 + [4.0] * 3)
    estimator = UKF(_satellite(), state, dt=1.0)
    gamma, mean_weights, covariance_weights = estimator._weights(state)
    np.testing.assert_allclose(
        covariance_weights[0] - mean_weights[0], 1 - gamma**2 / 6 + estimator.beta
    )


def test_ukf_correction_reduces_attitude_error(orbital_state):
    estimator = UKF(_satellite(), _state(), dt=0.1)
    prior = estimator.state
    truth = prior.plus([0.0, 0.0, 0.0, 0.05, -0.025, 0.01])

    corrected = estimator.correct(truth.q, orbital_state)

    assert np.linalg.norm(truth.minus(corrected)) < np.linalg.norm(truth.minus(prior))
    assert np.isclose(np.linalg.norm(corrected.q), 1.0)
    assert np.linalg.eigvalsh(corrected.cov).min() >= -1.0e-12


def test_ukf_update_corrects_the_first_sample_then_steps(orbital_state):
    estimator = UKF(_satellite(), _state(), dt=0.1)
    measurements = estimator.satellite.measurement_stack.predict(estimator.state, orbital_state)

    estimator.update(np.empty(0), measurements, orbital_state)
    assert "process_noise" not in estimator.diagnostics

    estimator.update(np.empty(0), measurements, orbital_state)
    assert estimator.diagnostics["process_noise"].shape == (6, 6)


def test_ukf_staged_predict_step_update_contract(orbital_state):
    estimator = UKF(_satellite(), _state(), dt=0.1)
    measurements = estimator.satellite.measurement_stack.predict(
        estimator.state, orbital_state
    )

    predicted = estimator.predict(
        np.empty(0), orbital_state, orbital_state,
        midpoint_orbital_state=orbital_state,
    )
    corrected = estimator.step(measurements, orbital_state)
    committed = estimator.update()

    assert predicted.covariance.shape == (6, 6)
    assert corrected.covariance.shape == (6, 6)
    np.testing.assert_allclose(
        committed.as_estimator_array(), corrected.as_estimator_array()
    )


def test_ukf_rejects_augmented_and_invalid_unscented_layouts():
    satellite = EstimatedSatellite(
        sensors=[
            Gyro(
                axis=np.array([1.0, 0.0, 0.0]),
                noise=Noise(std_noise=1.0e-3),
                estimate_bias=True,
            )
        ]
    )
    augmented = EstimatorState(
        w=np.zeros(3),
        q=[1.0, 0.0, 0.0, 0.0],
        sens_bias=[0.0],
        cov=np.eye(7),
        int_cov=np.zeros((7, 7)),
    )

    with pytest.raises(NotImplementedError, match="does not yet support estimated biases"):
        UKF(satellite, augmented, dt=0.1)
    with pytest.raises(ValueError, match="positive UKF scale"):
        UKF(_satellite(), _state(), dt=0.1, kappa=-6.0)


def test_srukf_converts_and_retains_square_root_covariance(orbital_state):
    estimator = SRUKF(
        _satellite(), _state(), dt=0.1, unmodeled_dynamics_psd=1.0e-9
    )
    assert estimator.state.covariance.form == "sqrt"
    assert estimator.state.process_noise.form == "sqrt"

    predicted = estimator.predict(np.empty(0), orbital_state, orbital_state)
    measurements = estimator.satellite.measurement_stack.predict(predicted, orbital_state)
    corrected = estimator.correct(measurements, orbital_state)

    assert corrected.covariance.form == "sqrt"
    assert corrected.process_noise.form == "sqrt"
    assert np.linalg.eigvalsh(corrected.cov).min() >= -1.0e-12


@pytest.mark.parametrize("filter_type", [UKF, SRUKF])
def test_perfect_actuators_do_not_augment_prediction_sigma_points(
    filter_type, orbital_state
):
    estimator = filter_type(_actuated_satellite([0.0, 0.0, 0.0]), _state(), dt=0.1)
    control = np.array([0.2, -0.1, 0.05])

    estimator.predict(control, orbital_state, orbital_state)
    diagnostics = estimator.diagnostics

    assert diagnostics["sigma_weights_mean"].shape == (13,)
    assert diagnostics["active_control_indices"].size == 0
    assert diagnostics["control_noise_offsets"].shape == (0, 0)
    np.testing.assert_allclose(
        diagnostics["prediction_sigma_controls"],
        np.repeat(control[None, :], 13, axis=0),
    )


@pytest.mark.parametrize("filter_type", [UKF, SRUKF])
def test_noisy_actuators_are_augmented_and_propagated_through_control_sigma_points(
    filter_type, orbital_state
):
    estimator = filter_type(_actuated_satellite([0.03, 0.02, 0.01]), _state(), dt=0.1)
    control = np.array([0.2, -0.1, 0.05])

    estimator.predict(control, orbital_state, orbital_state)
    diagnostics = estimator.diagnostics
    sigma_controls = diagnostics["prediction_sigma_controls"]

    assert diagnostics["sigma_weights_mean"].shape == (19,)
    np.testing.assert_array_equal(diagnostics["active_control_indices"], [0, 1, 2])
    np.testing.assert_allclose(
        diagnostics["control_noise_covariance"],
        np.diag(np.square([0.03, 0.02, 0.01])),
    )
    assert np.any(np.abs(sigma_controls[13:] - control) > 0.0)
    assert np.any(
        np.abs(diagnostics["predicted_sigma_deviations"][13:]) > 1.0e-14
    )
    assert np.linalg.eigvalsh(estimator.state.cov).min() >= -1.0e-12


@pytest.mark.parametrize("filter_type", [UKF, SRUKF])
def test_only_noisy_actuator_channels_expand_the_prediction(filter_type, orbital_state):
    estimator = filter_type(_actuated_satellite([0.0, 0.02, 0.0]), _state(), dt=0.1)
    control = np.array([0.2, -0.1, 0.05])

    estimator.predict(control, orbital_state, orbital_state)
    diagnostics = estimator.diagnostics
    sigma_controls = diagnostics["prediction_sigma_controls"]

    # Six tangent-state dimensions plus precisely one noisy command channel.
    assert diagnostics["sigma_weights_mean"].shape == (15,)
    np.testing.assert_array_equal(diagnostics["active_control_indices"], [1])
    np.testing.assert_allclose(
        sigma_controls[13:, [0, 2]],
        np.repeat(control[[0, 2]][None, :], 2, axis=0),
    )
    assert np.any(np.abs(sigma_controls[13:, 1] - control[1]) > 0.0)
