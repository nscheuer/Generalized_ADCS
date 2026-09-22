"""Focused checks for the architecture prototype."""

from dataclasses import replace

import numpy as np
import pytest

from .prototype import (
    AsyncEventEngine,
    FixedStepEngine,
    ModelConfig,
    Timing,
    build_satellite,
    default_timings,
)


def test_graph_declares_and_connects_typed_ports():
    graph = build_satellite(ModelConfig(duration=0.2))
    assert ("sensors", "estimator", "sensors") in graph.connections
    assert ("controller", "actuators", "control") in graph.connections
    assert ("actuators", "attitude", "actuator") in graph.connections


@pytest.mark.parametrize("orbit_degree,orbit_scheme,attitude_scheme", [(2, "rk2", "rk2"), (4, "rk4", "rk4")])
def test_fixed_backend_integrator_choices_are_finite(orbit_degree, orbit_scheme, attitude_scheme):
    config = ModelConfig(
        duration=0.4,
        orbit_degree=orbit_degree,
        orbit_integrator=orbit_scheme,
        attitude_integrator=attitude_scheme,
    )
    result = FixedStepEngine().run(build_satellite(config))
    attitude = result.array("attitude")
    assert result.base_tick == 0.02
    assert np.isfinite(attitude).all()
    np.testing.assert_allclose(np.linalg.norm(attitude[:, 3:7], axis=1), 1.0, atol=1e-12)


def test_async_backend_delivers_outputs_after_declared_delays():
    config = ModelConfig(duration=0.5, timings=default_timings(asynchronous=True))
    result = AsyncEventEngine().run(build_satellite(config))
    sensor_events = [entry for entry in result.trace if entry.block == "sensors"]
    assert sensor_events
    assert all(entry.delivery > entry.start for entry in sensor_events)
    assert max(result.history["sensors_age"]) > config.timings["sensors"].latency


def test_fixed_backend_rejects_off_lattice_clock():
    timings = default_timings()
    timings["sensors"] = Timing(period=0.11)
    config = replace(ModelConfig(duration=0.2), timings=timings)
    with pytest.raises(ValueError, match="lattice"):
        FixedStepEngine().run(build_satellite(config))
