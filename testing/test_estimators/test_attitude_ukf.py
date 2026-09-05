"""Contracts for the non-augmented tangent-space attitude UKF."""

from __future__ import annotations

import numpy as np
import pytest

from ADCS.estimators.attitude_estimators import SRUKF, UKF
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.errors import Noise
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
    state.cov[3:6, 3:6] = np.eye(3)
    estimator = UKF(_satellite(), state, dt=0.1)

    estimator.predict(np.empty(0), orbital_state, orbital_state)

    attitude_offsets = estimator.diagnostics["sigma_offsets"][:, 3:6]
    assert np.linalg.norm(attitude_offsets, axis=1).max() <= 1.9 + 1.0e-12


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
