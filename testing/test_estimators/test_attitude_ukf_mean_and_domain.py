"""UKF manifold-mean and chart-domain regressions from the September 2026 estimator audit."""
from __future__ import annotations

import numpy as np
import pytest

from ADCS.estimators.attitude_estimators import UKF
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.errors import Noise
from ADCS.satellite_hardware.satellite import EstimatedSatellite
from ADCS.satellite_hardware.sensors import StarTrackerQuaternion
from ADCS.state import EstimatorState


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


def _mean_residual(estimator: UKF) -> float:
    d = estimator.diagnostics
    return float(np.linalg.norm(d["sigma_weights_mean"] @ d["predicted_sigma_deviations"]))


@pytest.mark.parametrize("alpha", [1.0e-3, 1.0e-4])
@pytest.mark.parametrize("attitude_variance", [0.1, 1.0])
def test_small_alpha_state_mean_converges(orbital_state, alpha, attitude_variance):
    estimator = UKF(_satellite(), _state(attitude_variance), dt=0.1, alpha=alpha)
    estimator.predict(np.empty(0), orbital_state, orbital_state)
    assert _mean_residual(estimator) < 1.0e-9


def test_cayley_chart_converges_at_unit_attitude_variance(orbital_state):
    estimator = UKF(_satellite(), _state(1.0), dt=0.1, quaternion_mode="cayley")
    estimator.predict(np.empty(0), orbital_state, orbital_state)
    assert _mean_residual(estimator) < 1.0e-12


@pytest.mark.parametrize("sigma", [1.41, 2.0, 3.0])
def test_quaternion_vector_survives_large_attitude_uncertainty(orbital_state, sigma):
    estimator = UKF(_satellite(), _state(sigma**2), dt=0.1)
    estimator.predict(np.empty(0), orbital_state, orbital_state)
    assert _mean_residual(estimator) < 1.0e-12
    truth = estimator.state.plus([0.0, 0.0, 0.0, 0.05, -0.025, 0.01])
    corrected = estimator.correct(truth.q, orbital_state)
    assert np.isfinite(corrected.cov).all()


def test_covariance_weights_use_the_effective_alpha_after_the_clamp(orbital_state):
    estimator = UKF(_satellite(), _state(1.0), dt=0.1)  # clamp active: gamma = 1.9 < sqrt(6)
    estimator.predict(np.empty(0), orbital_state, orbital_state)
    d = estimator.diagnostics
    gamma = np.linalg.norm(d["sigma_offsets"][3, 3:6])
    assert gamma == pytest.approx(1.9, abs=1e-12)
    effective_alpha_sq = gamma**2 / 6.0
    assert d["sigma_weights_covariance"].sum() == pytest.approx(1.0 + 1.0 - effective_alpha_sq + 2.0, abs=1e-12)
    # equivalent to a filter constructed with the effective alpha
    twin = UKF(_satellite(), _state(1.0), dt=0.1, alpha=np.sqrt(effective_alpha_sq))
    twin_prediction = twin.predict(np.empty(0), orbital_state, orbital_state)
    np.testing.assert_allclose(twin_prediction.cov, estimator.state.cov, atol=1e-15)
