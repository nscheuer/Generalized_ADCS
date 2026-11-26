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


def test_keplerian_orbital_propagator_contract(
    propagation_backend: SimpleNamespace,
) -> None:
    state, _sat, env, dyn = _setup(propagation_backend)
    prop = propagation_backend.propagation.KeplerianOrbitalPropagator(0.5)
    assert prop.dt == pytest.approx(0.5)
    stepped = prop.step(state, env, dyn)
    assert stepped.orbit.position[1] > 0.0
    times, states = prop.integrate(state, env, dyn, tf=1.0)
    np.testing.assert_allclose(times, [0.0, 0.5, 1.0])
    assert states.shape == (3, 13)
    with pytest.raises(ValueError):
        propagation_backend.propagation.KeplerianOrbitalPropagator(0.0)


def test_rk4_attitude_propagator_contract(
    propagation_backend: SimpleNamespace,
) -> None:
    prop = propagation_backend.propagation.RK4AttitudePropagator(0.5)
    assert prop.dt == pytest.approx(0.5)
    with pytest.raises(ValueError):
        propagation_backend.propagation.RK4AttitudePropagator(0.0)


def test_reference_attitude_propagator_contract(
    propagation_backend: SimpleNamespace,
) -> None:
    nadir = propagation_backend.propagation.ReferenceAttitudePropagator(0.5, "nadir")
    assert nadir.dt == pytest.approx(0.5)
    along = propagation_backend.propagation.ReferenceAttitudePropagator(
        1.0, "along-track"
    )
    assert along.dt == pytest.approx(1.0)
    with pytest.raises(ValueError):
        propagation_backend.propagation.ReferenceAttitudePropagator(0.0, "nadir")
    with pytest.raises(ValueError):
        propagation_backend.propagation.ReferenceAttitudePropagator(0.5, "invalid")


def test_coupled_propagators_contract(
    propagation_backend: SimpleNamespace,
) -> None:
    state, sat, env, dyn = _setup(propagation_backend)
    for prop in (
        propagation_backend.propagation.CoupledRK4Propagator(0.5),
        propagation_backend.propagation.CoupledDOP853Propagator(
            0.5, rtol=1e-9, atol=1e-12
        ),
    ):
        assert prop.dt == pytest.approx(0.5)
        stepped = prop.step(state, sat, env, dyn)
        assert stepped.orbit.position[1] > 0.0
        times, states = prop.integrate(state, sat, env, dyn, tf=1.0)
        assert times[0] == 0.0
        assert times[-1] == pytest.approx(1.0)
        assert states.shape[1] == 13
