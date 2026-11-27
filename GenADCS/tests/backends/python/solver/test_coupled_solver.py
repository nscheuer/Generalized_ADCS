import math
import numpy as np
import pytest

from GenADCS.backends.python.config import DynamicsConfig
from GenADCS.backends.python.environment import Environment
from GenADCS.backends.python.hardware import Satellite, SatelliteGeometry
from GenADCS.backends.python.propagation import CoupledRK4Propagator
from GenADCS.backends.python.solver import CoupledSolver
from GenADCS.backends.python.state import AttitudeState, OrbitalState, State


def test_coupled_solver_attributes_and_step() -> None:
    propagator = CoupledRK4Propagator(0.25)
    solver = CoupledSolver(propagator, tf=10.0, progress=True, progress_chunk_steps=20)
    assert solver.propagator is propagator
    assert solver.tf == 10.0
    assert solver.progress is True
    assert solver.progress_chunk_steps == 20
    assert solver.dt == 0.25

    state = State(
        AttitudeState(np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.1])),
        OrbitalState.from_cartesian([7e6, 0.0, 0.0], [0.0, 7500.0, 0.0], 2451545.0),
    )
    satellite = Satellite(SatelliteGeometry(10.0, np.eye(3)))
    next_state = solver.step(state, satellite, Environment(), DynamicsConfig())
    np.testing.assert_allclose(
        next_state.attitude.quaternion_array,
        [np.cos(0.0125), 0.0, 0.0, -np.sin(0.0125)],
        atol=1e-9,
    )
    assert next_state.orbit.position[1] > 0.0


def test_coupled_solver_defaults_and_validation() -> None:
    zero_duration = CoupledSolver(CoupledRK4Propagator(1.0), tf=0.0)
    assert zero_duration.tf == 0.0
    assert zero_duration.progress is True
    assert zero_duration.progress_chunk_steps == 500

    for bad_tf in (-1.0, math.inf, math.nan):
        with pytest.raises(ValueError, match="nonnegative"):
            CoupledSolver(CoupledRK4Propagator(0.1), tf=bad_tf)

    for bad_chunk in (0, -1, False):
        with pytest.raises(ValueError, match="positive integer"):
            CoupledSolver(CoupledRK4Propagator(0.1), tf=1.0, progress_chunk_steps=bad_chunk)
