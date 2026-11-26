"""Python-facing integration checks for compiled propagation backends."""

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
        state=import_module(f"GenADCS.backends.{name}.state"),
    )


def _make_setup(backend: SimpleNamespace) -> tuple[Any, Any, Any, Any]:
    env = backend.environment.Environment()
    mu = env.planetary_body.gravitational_parameter
    radius = 7_000_000.0
    speed = float(np.sqrt(mu / radius))
    state = backend.state.State(
        backend.state.AttitudeState(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 0.1], dtype=np.float32),
        ),
        backend.state.OrbitalState.from_cartesian(
            [radius, 0.0, 0.0], [0.0, speed, 0.0], 2451545.0
        ),
    )
    sat = backend.hardware.Satellite(
        backend.hardware.SatelliteGeometry(10.0, np.eye(3, dtype=np.float32))
    )
    dyn = backend.config.DynamicsConfig()
    return state, sat, env, dyn


def test_compiled_propagators_marshal_arrays_and_return_float64(
    backend: SimpleNamespace,
) -> None:
    state, sat, env, dyn = _make_setup(backend)

    kep = backend.propagation.KeplerianOrbitalPropagator(0.5)
    kep_times, kep_states = kep.integrate(state, env, dyn, tf=1.0)
    assert kep_times.dtype == np.float64
    assert kep_states.dtype == np.float64
    assert kep_states.shape == (3, 13)

    rk4_att = backend.propagation.RK4AttitudePropagator(0.5)
    assert rk4_att.dt == pytest.approx(0.5)

    coupled_rk4 = backend.propagation.CoupledRK4Propagator(0.5)
    crk4_times, crk4_states = coupled_rk4.integrate(state, sat, env, dyn, tf=1.0)
    assert crk4_times.dtype == np.float64
    assert crk4_states.dtype == np.float64
    assert crk4_states.shape == (3, 13)

    coupled_dop = backend.propagation.CoupledDOP853Propagator(
        0.5, rtol=1e-9, atol=np.full(13, 1e-12, dtype=np.float32)
    )
    dop_times, dop_states = coupled_dop.integrate(state, sat, env, dyn, tf=1.0)
    assert dop_times.dtype == np.float64
    assert dop_states.dtype == np.float64
    assert dop_states.shape[1] == 13


def test_compiled_propagators_map_validation_errors_to_value_error(
    backend: SimpleNamespace,
) -> None:
    state, sat, env, dyn = _make_setup(backend)

    for cls in (
        backend.propagation.KeplerianOrbitalPropagator,
        backend.propagation.RK4AttitudePropagator,
        backend.propagation.CoupledRK4Propagator,
        backend.propagation.CoupledDOP853Propagator,
    ):
        with pytest.raises(ValueError):
            cls(0.0)
        with pytest.raises(ValueError):
            cls(-1.0)

    with pytest.raises(ValueError):
        backend.propagation.ReferenceAttitudePropagator(0.0, "nadir")
    with pytest.raises(ValueError):
        backend.propagation.ReferenceAttitudePropagator(0.5, "invalid")
    with pytest.raises(ValueError):
        backend.propagation.CoupledDOP853Propagator(0.5, atol=np.ones(4))
    with pytest.raises(ValueError):
        backend.propagation.CoupledRK4Propagator(0.5).integrate(
            state, sat, env, dyn, tf=-0.1
        )
