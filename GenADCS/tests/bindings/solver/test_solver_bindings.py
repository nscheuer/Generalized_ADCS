"""Python-facing integration checks for compiled solver backends."""

import math
from importlib import import_module
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


@pytest.fixture(params=("cpp", "rust"))
def backend(request: pytest.FixtureRequest) -> SimpleNamespace:
    name = request.param
    return SimpleNamespace(
        name=name,
        config=import_module(f"GenADCS.backends.{name}.config"),
        environment=import_module(f"GenADCS.backends.{name}.environment"),
        hardware=import_module(f"GenADCS.backends.{name}.hardware"),
        propagation=import_module(f"GenADCS.backends.{name}.propagation"),
        solver=import_module(f"GenADCS.backends.{name}.solver"),
        state=import_module(f"GenADCS.backends.{name}.state"),
    )


def test_compiled_solver_bindings_marshal_and_validate(
    backend: SimpleNamespace,
) -> None:
    orbital = backend.solver.OrbitalSolver(
        backend.propagation.KeplerianOrbitalPropagator(0.25)
    )
    attitude = backend.solver.AttitudeSolver(
        backend.propagation.RK4AttitudePropagator(0.25)
    )
    assert orbital.dt == pytest.approx(0.25)
    assert attitude.dt == pytest.approx(0.25)

    split = backend.solver.Solver(
        dynamics=backend.config.DynamicsConfig(),
        orbital=orbital,
        attitude=attitude,
        tf=0.5,
    )
    assert split.tf == pytest.approx(0.5)
    assert split.coupled is None

    coupled_inner = backend.solver.CoupledSolver(
        backend.propagation.CoupledRK4Propagator(0.25),
        tf=0.5,
        progress=False,
        progress_chunk_steps=10,
    )
    assert coupled_inner.dt == pytest.approx(0.25)
    assert coupled_inner.tf == pytest.approx(0.5)
    assert coupled_inner.progress_chunk_steps == 10

    coupled = backend.solver.Solver(
        dynamics=backend.config.DynamicsConfig(),
        coupled=coupled_inner,
    )
    assert coupled.coupled is not None

    state = backend.state.State(
        backend.state.AttitudeState([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.1]),
        backend.state.OrbitalState.from_cartesian(
            [7e6, 0.0, 0.0], [0.0, 7500.0, 0.0], 2451545.0
        ),
    )
    sat = backend.hardware.Satellite(
        backend.hardware.SatelliteGeometry(10.0, np.eye(3))
    )
    env = backend.environment.Environment()
    dyn = backend.config.DynamicsConfig()
    next_state = coupled_inner.step(state, sat, env, dyn)
    assert next_state.orbit.position[1] > 0.0

    with pytest.raises(ValueError):
        backend.solver.Solver()
    with pytest.raises(ValueError):
        backend.solver.Solver(orbital=orbital, tf=1.0)
    with pytest.raises(ValueError):
        backend.solver.CoupledSolver(
            backend.propagation.CoupledRK4Propagator(0.25), tf=-1.0
        )
    with pytest.raises(ValueError):
        backend.solver.CoupledSolver(
            backend.propagation.CoupledRK4Propagator(0.25), tf=math.nan
        )
    with pytest.raises(ValueError):
        backend.solver.CoupledSolver(
            backend.propagation.CoupledRK4Propagator(0.25),
            tf=1.0,
            progress_chunk_steps=0,
        )
