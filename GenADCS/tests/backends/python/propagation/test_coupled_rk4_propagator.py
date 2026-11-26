import numpy as np
import pytest

from GenADCS.backends.python.config import DynamicsConfig
from GenADCS.backends.python.environment import Environment
from GenADCS.backends.python.hardware import Satellite, SatelliteGeometry
from GenADCS.backends.python.propagation import CoupledRK4Propagator
from GenADCS.backends.python.propagation.coupled_propagators.coupled_rk4_propagator import (
    _rk4_step,
)
from GenADCS.backends.python.state import AttitudeState, OrbitalState, State


def test_coupled_rk4_step_rotates_and_advances_orbit() -> None:
    environment = Environment()
    mu = environment.planetary_body.gravitational_parameter
    radius = 7_000_000.0
    speed = np.sqrt(mu / radius)
    state = State(
        AttitudeState(np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.1])),
        OrbitalState.from_cartesian([radius, 0.0, 0.0], [0.0, speed, 0.0], 2451545.0),
    )
    satellite = Satellite(SatelliteGeometry(10.0, np.eye(3)))
    next_state = CoupledRK4Propagator(1.0).step(
        state, satellite, environment, DynamicsConfig()
    )
    np.testing.assert_allclose(
        next_state.attitude.quaternion_array,
        [np.cos(0.05), 0.0, 0.0, -np.sin(0.05)],
        atol=1e-8,
    )
    np.testing.assert_allclose(next_state.attitude.rates, [0.0, 0.0, 0.1], atol=1e-12)
    assert next_state.orbit.position[1] > 0.0
    assert next_state.orbit.julian_date == pytest.approx(2451545.0 + 1.0 / 86400.0)
    assert _rk4_step.nopython_signatures


def test_coupled_rk4_integrate_short_horizon_and_validation() -> None:
    environment = Environment()
    mu = environment.planetary_body.gravitational_parameter
    radius = 7_000_000.0
    state = State(
        AttitudeState(np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.1])),
        OrbitalState.from_cartesian(
            [radius, 0.0, 0.0], [0.0, np.sqrt(mu / radius), 0.0], 2451545.0
        ),
    )
    satellite = Satellite(SatelliteGeometry(10.0, np.eye(3)))
    times, states = CoupledRK4Propagator(0.25).integrate(
        state, satellite, environment, DynamicsConfig(), tf=1.1, record_every=3
    )
    np.testing.assert_allclose(times, [0.0, 0.75, 1.1])
    np.testing.assert_allclose(
        states[-1, :4], [np.cos(0.055), 0.0, 0.0, -np.sin(0.055)], atol=1e-9
    )
    assert states[-1, 8] > 0.0

    with pytest.raises(ValueError, match="dt must be finite and positive"):
        CoupledRK4Propagator(0.0)
    with pytest.raises(ValueError, match="nonnegative"):
        CoupledRK4Propagator(0.25).integrate(
            state, satellite, environment, DynamicsConfig(), tf=-0.1
        )
