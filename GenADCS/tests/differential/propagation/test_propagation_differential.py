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


def test_keplerian_orbital_propagator_differential(
    propagation_backends: dict[str, Any],
) -> None:
    results = {}
    for name, backend in propagation_backends.items():
        state, _sat, env, dyn = _setup(backend)
        results[name] = backend.propagation.KeplerianOrbitalPropagator(0.25).integrate(
            state, env, dyn, tf=1.0
        )
    ref_times, ref_states = results["python"]
    for name in ("cpp", "rust"):
        times, states = results[name]
        np.testing.assert_allclose(times, ref_times, rtol=0, atol=1e-12)
        np.testing.assert_allclose(states, ref_states, rtol=1e-12, atol=1e-10)


def test_rk4_attitude_propagator_differential(
    propagation_backends: dict[str, Any],
) -> None:
    results = {}
    for name in ("python", "cpp"):
        backend = propagation_backends[name]
        state, sat, env, dyn = _setup(backend)
        results[name] = backend.propagation.RK4AttitudePropagator(0.25).integrate(
            state, sat, env, dyn, tf=1.0
        )
    ref_times, ref_states = results["python"]
    times, states = results["cpp"]
    np.testing.assert_allclose(times, ref_times, rtol=0, atol=1e-12)
    np.testing.assert_allclose(states, ref_states, rtol=1e-12, atol=1e-10)


def test_coupled_rk4_propagator_differential(
    propagation_backends: dict[str, Any],
) -> None:
    results = {}
    for name, backend in propagation_backends.items():
        state, sat, env, dyn = _setup(backend)
        results[name] = backend.propagation.CoupledRK4Propagator(0.25).integrate(
            state, sat, env, dyn, tf=1.0
        )
    ref_times, ref_states = results["python"]
    for name in ("cpp", "rust"):
        times, states = results[name]
        np.testing.assert_allclose(times, ref_times, rtol=0, atol=1e-12)
        np.testing.assert_allclose(states, ref_states, rtol=1e-11, atol=1e-9)


def test_coupled_dop853_propagator_differential(
    propagation_backends: dict[str, Any],
) -> None:
    results = {}
    for name, backend in propagation_backends.items():
        state, sat, env, dyn = _setup(backend)
        results[name] = backend.propagation.CoupledDOP853Propagator(
            0.5, rtol=1e-10, atol=1e-12
        ).integrate(state, sat, env, dyn, tf=1.0)
    ref_times, ref_states = results["python"]
    for name in ("cpp", "rust"):
        times, states = results[name]
        np.testing.assert_allclose(times[-1], ref_times[-1], rtol=0, atol=1e-12)
        np.testing.assert_allclose(states[-1], ref_states[-1], rtol=1e-9, atol=1e-9)
