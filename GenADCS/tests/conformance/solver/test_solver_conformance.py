from types import SimpleNamespace

import numpy as np
import pytest


def _setup(backend: SimpleNamespace):
    env = backend.environment.Environment()
    mu = env.planetary_body.gravitational_parameter
    radius = 7_000_000.0
    speed = float(np.sqrt(mu / radius))
    state = backend.state.State(
        backend.state.AttitudeState([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.1]),
        backend.state.OrbitalState.from_cartesian(
            [radius, 0.0, 0.0], [0.0, speed, 0.0], 2451545.0
        ),
    )
    sat = backend.hardware.Satellite(
        backend.hardware.SatelliteGeometry(10.0, np.eye(3))
    )
    dyn = backend.config.DynamicsConfig()
    return state, sat, env, dyn


def test_coupled_and_uncoupled_simulate_contract(
    solver_backend: SimpleNamespace,
) -> None:
    state, sat, env, dyn = _setup(solver_backend)

    coupled_solver = solver_backend.solver.Solver(
        dynamics=dyn,
        coupled=solver_backend.solver.CoupledSolver(
            solver_backend.propagation.CoupledRK4Propagator(0.25),
            tf=0.5,
        ),
    )
    fmt = solver_backend.ResultsFormat(
        satellite_position=True, satellite_velocity=True
    )
    coupled_res = solver_backend.simulate(
        sat, env, dyn, state, coupled_solver, resultsformat=fmt
    )
    coupled_samples = coupled_res.satellites["sat_000"]
    np.testing.assert_allclose(coupled_samples.times_s, [0.0, 0.25, 0.5])
    assert coupled_samples.fields["satellite_position"][-1, 1] > 0.0

    split_solver = solver_backend.solver.Solver(
        dynamics=dyn,
        orbital=solver_backend.solver.OrbitalSolver(
            solver_backend.propagation.KeplerianOrbitalPropagator(0.25)
        ),
        attitude=solver_backend.solver.AttitudeSolver(
            solver_backend.propagation.RK4AttitudePropagator(0.25)
        ),
        tf=0.5,
    )
    split_res = solver_backend.simulate(
        sat, env, dyn, state, split_solver, resultsformat=fmt
    )
    split_samples = split_res.satellites["sat_000"]
    np.testing.assert_allclose(split_samples.times_s, [0.0, 0.25, 0.5])
    assert split_samples.fields["satellite_position"][-1, 1] > 0.0


def test_solver_validation_contract(solver_backend: SimpleNamespace) -> None:
    orbital = solver_backend.solver.OrbitalSolver(
        solver_backend.propagation.KeplerianOrbitalPropagator(0.25)
    )
    with pytest.raises(ValueError):
        solver_backend.solver.Solver(orbital=orbital, tf=1.0)
    with pytest.raises(ValueError):
        solver_backend.solver.Solver()
    with pytest.raises(ValueError):
        solver_backend.solver.CoupledSolver(
            solver_backend.propagation.CoupledRK4Propagator(0.25), tf=-1.0
        )
