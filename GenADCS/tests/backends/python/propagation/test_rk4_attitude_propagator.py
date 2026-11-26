import numpy as np
import pytest

from GenADCS.backends.python.config import DynamicsConfig
from GenADCS.backends.python.environment import Environment
from GenADCS.backends.python.hardware import Satellite, SatelliteGeometry
from GenADCS.backends.python.propagation import RK4AttitudePropagator
from GenADCS.backends.python.propagation.attitude_propagators.rk4_attitude_propagator import (
    _rk4_attitude_step,
)
from GenADCS.backends.python.state import AttitudeState, OrbitalState, State


def test_torque_free_rotation_preserves_orbit_and_rate() -> None:
    state = State(
        AttitudeState(np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.1])),
        OrbitalState.from_cartesian([7e6, 0, 0], [0, 7500, 0], 2451545.0),
    )
    satellite = Satellite(SatelliteGeometry(10.0, np.eye(3)))
    result = RK4AttitudePropagator(1.0).step(
        state, satellite, Environment(), DynamicsConfig()
    )
    np.testing.assert_allclose(
        result.attitude.quaternion_array,
        [np.cos(0.05), 0.0, 0.0, -np.sin(0.05)],
        atol=1e-8,
    )
    np.testing.assert_allclose(result.attitude.rates, state.attitude.rates)
    assert result.orbit is state.orbit
    assert _rk4_attitude_step.nopython_signatures


def test_invalid_step_duration_rejected() -> None:
    with pytest.raises(ValueError, match="dt must be finite and positive"):
        RK4AttitudePropagator(0.0)


def test_multiple_steps_stay_in_one_numba_integration() -> None:
    from GenADCS.backends.python.propagation.attitude_propagators.rk4_attitude_propagator import (
        _integrate_attitude,
    )

    state = State(
        AttitudeState(np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.1])),
        OrbitalState.from_cartesian([7e6, 0, 0], [0, 7500, 0], 2451545.0),
    )
    satellite = Satellite(SatelliteGeometry(10.0, np.eye(3)))
    times, states = RK4AttitudePropagator(0.25).integrate(
        state, satellite, Environment(), DynamicsConfig(), tf=1.1, record_every=3
    )
    np.testing.assert_allclose(times, [0.0, 0.75, 1.1])
    np.testing.assert_allclose(
        states[-1, :4], [np.cos(0.055), 0.0, 0.0, -np.sin(0.055)], atol=1e-9
    )
    np.testing.assert_allclose(states[:, 7:13], np.tile(states[0, 7:13], (3, 1)))
    assert _integrate_attitude.nopython_signatures
