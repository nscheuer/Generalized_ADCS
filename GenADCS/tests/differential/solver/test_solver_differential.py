from types import SimpleNamespace
from typing import Any

import numpy as np


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


def _assert_results_match(results: dict[str, Any]) -> None:
    ref = results["python"].satellites["sat_000"]
    for name in ("cpp", "rust"):
        samples = results[name].satellites["sat_000"]
        np.testing.assert_allclose(samples.times_s, ref.times_s, rtol=0, atol=1e-12)
        ref_q = ref.fields["satellite_attitude"]
        act_q = samples.fields["satellite_attitude"]
        signs = np.where(np.sum(ref_q * act_q, axis=1) < 0.0, -1.0, 1.0)[:, None]
        np.testing.assert_allclose(
            act_q * signs, ref_q, rtol=1e-10, atol=1e-9
        )
        for field in (
            "satellite_rate",
            "satellite_position",
            "satellite_velocity",
        ):
            np.testing.assert_allclose(
                samples.fields[field], ref.fields[field], rtol=1e-10, atol=1e-9
            )


def test_coupled_rk4_solver_differential(solver_backends: dict[str, Any]) -> None:
    results = {}
    for name, backend in solver_backends.items():
        state, sat, env, dyn = _setup(backend)
        solver = backend.solver.Solver(
            dynamics=dyn,
            coupled=backend.solver.CoupledSolver(
                backend.propagation.CoupledRK4Propagator(0.25),
                tf=0.5,
            ),
        )
        fmt = backend.ResultsFormat(satellite_position=True, satellite_velocity=True)
        results[name] = backend.simulate(sat, env, dyn, state, solver, resultsformat=fmt)
    _assert_results_match(results)


def test_coupled_dop853_solver_differential(solver_backends: dict[str, Any]) -> None:
    results = {}
    for name, backend in solver_backends.items():
        state, sat, env, dyn = _setup(backend)
        solver = backend.solver.Solver(
            dynamics=dyn,
            coupled=backend.solver.CoupledSolver(
                backend.propagation.CoupledDOP853Propagator(
                    0.25, rtol=1e-10, atol=1e-12
                ),
                tf=0.5,
            ),
        )
        fmt = backend.ResultsFormat(satellite_position=True, satellite_velocity=True)
        results[name] = backend.simulate(sat, env, dyn, state, solver, resultsformat=fmt)
    ref = results["python"].satellites["sat_000"]
    for name in ("cpp", "rust"):
        samples = results[name].satellites["sat_000"]
        np.testing.assert_allclose(
            samples.times_s[-1], ref.times_s[-1], rtol=0, atol=1e-12
        )
        for field in (
            "satellite_attitude",
            "satellite_rate",
            "satellite_position",
            "satellite_velocity",
        ):
            np.testing.assert_allclose(
                samples.fields[field][-1],
                ref.fields[field][-1],
                rtol=1e-9,
                atol=1e-9,
            )


def test_uncoupled_rk4_solver_differential(solver_backends: dict[str, Any]) -> None:
    results = {}
    for name, backend in solver_backends.items():
        state, sat, env, dyn = _setup(backend)
        solver = backend.solver.Solver(
            dynamics=dyn,
            orbital=backend.solver.OrbitalSolver(
                backend.propagation.KeplerianOrbitalPropagator(0.25)
            ),
            attitude=backend.solver.AttitudeSolver(
                backend.propagation.RK4AttitudePropagator(0.25)
            ),
            tf=0.5,
        )
        fmt = backend.ResultsFormat(satellite_position=True, satellite_velocity=True)
        results[name] = backend.simulate(sat, env, dyn, state, solver, resultsformat=fmt)
    _assert_results_match(results)


def test_uncoupled_reference_nadir_solver_differential(
    solver_backends: dict[str, Any],
) -> None:
    results = {}
    for name, backend in solver_backends.items():
        state, sat, env, dyn = _setup(backend)
        solver = backend.solver.Solver(
            dynamics=dyn,
            orbital=backend.solver.OrbitalSolver(
                backend.propagation.KeplerianOrbitalPropagator(0.25)
            ),
            attitude=backend.solver.AttitudeSolver(
                backend.propagation.ReferenceAttitudePropagator(0.25, "nadir")
            ),
            tf=0.5,
        )
        fmt = backend.ResultsFormat(satellite_position=True, satellite_velocity=True)
        results[name] = backend.simulate(sat, env, dyn, state, solver, resultsformat=fmt)
    _assert_results_match(results)
