import numpy as np
import pytest

from GenADCS.backends.python.config import DynamicsConfig
from GenADCS.backends.python.environment import Environment
from GenADCS.backends.python.propagation import KeplerianOrbitalPropagator
from GenADCS.backends.python.propagation.orbital_propagators.keplerian_orbital_propagator import (
    _integrate_keplerian,
    keplerian_step,
)
from GenADCS.backends.python.state import AttitudeState, OrbitalState, State


def test_circular_orbit_returns_after_one_period_in_numba() -> None:
    environment = Environment()
    mu = environment.planetary_body.gravitational_parameter
    radius = 7_000_000.0
    speed = np.sqrt(mu / radius)
    period = 2 * np.pi * np.sqrt(radius**3 / mu)
    state = State(
        AttitudeState(np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3)),
        OrbitalState.from_cartesian(
            np.array([radius, 0.0, 0.0]), np.array([0.0, speed, 0.0]), 2451545.0
        ),
    )

    result = KeplerianOrbitalPropagator(period).step(
        state, environment, DynamicsConfig()
    )

    np.testing.assert_allclose(result.orbit.position, state.orbit.position, atol=1e-6)
    np.testing.assert_allclose(result.orbit.velocity, state.orbit.velocity, atol=1e-8)
    assert result.orbit.julian_date == pytest.approx(2451545.0 + period / 86400)
    assert result.attitude is state.attitude
    assert result.internal is state.internal
    assert keplerian_step.nopython_signatures


def test_invalid_step_duration_rejected() -> None:
    with pytest.raises(ValueError, match="dt must be finite and positive"):
        KeplerianOrbitalPropagator(0.0)


def test_multiple_keplerian_steps_run_in_one_numba_call() -> None:
    environment = Environment()
    mu = environment.planetary_body.gravitational_parameter
    radius = 7_000_000.0
    speed = np.sqrt(mu / radius)
    state = State(
        AttitudeState(np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3)),
        OrbitalState.from_cartesian([radius, 0.0, 0.0], [0.0, speed, 0.0], 2451545.0),
    )
    times, states = KeplerianOrbitalPropagator(10.0).integrate(
        state, environment, DynamicsConfig(), tf=42.5, record_every=3
    )
    np.testing.assert_allclose(times, [0.0, 30.0, 42.5])
    np.testing.assert_allclose(
        states[-1, 7:13],
        np.concatenate(
            keplerian_step(state.orbit.position, state.orbit.velocity, mu, 42.5)
        ),
        atol=1e-6,
    )
    np.testing.assert_array_equal(states[:, :7], np.tile(states[0, :7], (3, 1)))
    assert _integrate_keplerian.nopython_signatures
