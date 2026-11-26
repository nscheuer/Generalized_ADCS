import numpy as np
import pytest

from GenADCS.backends.python.config import DynamicsConfig
from GenADCS.backends.python.environment import Environment
from GenADCS.backends.python.hardware import Satellite, SatelliteGeometry
from GenADCS.backends.python.propagation import CoupledDOP853Propagator
from GenADCS.backends.python.propagation.coupled_propagators.coupled_dop853_propagator import (
    _integrate_adaptive_chunk,
)
from GenADCS.backends.python.state import AttitudeState, OrbitalState, State


def test_coupled_dop853_step_and_integrate_short_horizon() -> None:
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
    propagator = CoupledDOP853Propagator(0.5, rtol=1e-9, atol=1e-12)

    stepped = propagator.step(state, satellite, environment, DynamicsConfig())
    np.testing.assert_allclose(
        stepped.attitude.quaternion_array,
        [np.cos(0.025), 0.0, 0.0, -np.sin(0.025)],
        atol=1e-10,
    )

    times, states = propagator.integrate(
        state, satellite, environment, DynamicsConfig(), tf=1.0
    )
    assert times[0] == 0.0
    assert times[-1] == pytest.approx(1.0)
    np.testing.assert_allclose(
        states[-1, :4], [np.cos(0.05), 0.0, 0.0, -np.sin(0.05)], atol=1e-10
    )
    assert states[-1, 8] > 0.0
    assert _integrate_adaptive_chunk.nopython_signatures


def test_coupled_dop853_validates_parameters() -> None:
    with pytest.raises(ValueError, match="dt must be finite and positive"):
        CoupledDOP853Propagator(0.0)
    with pytest.raises(ValueError, match="rtol must be finite and positive"):
        CoupledDOP853Propagator(0.5, rtol=0.0)
    with pytest.raises(ValueError, match="atol"):
        CoupledDOP853Propagator(0.5, atol=np.ones(5))
    with pytest.raises(ValueError, match="min_step"):
        CoupledDOP853Propagator(0.5, max_step=0.1, min_step=0.2)
