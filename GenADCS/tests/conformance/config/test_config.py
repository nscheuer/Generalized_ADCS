from types import SimpleNamespace
from typing import Any

import pytest


class State:
    def __init__(self) -> None:
        self.orbit = SimpleNamespace(julian_date=2451545.0)
        self.record = object()
        self.count = None

    def to_record(self, count: int) -> object:
        self.count = count
        return self.record


def dependencies(wheels: int = 2) -> tuple[Any, object, State, object, object]:
    return (
        SimpleNamespace(adcs=SimpleNamespace(reaction_wheels=[None] * wheels)),
        object(),
        State(),
        object(),
        object(),
    )


def test_dynamics_config_defaults_and_mutable_collections(config_api: Any) -> None:
    first, second = config_api.DynamicsConfig(), config_api.DynamicsConfig()
    assert first.forces == first.torques == []
    first.forces = ["gravity"]
    first.torques = ["control"]
    assert first.forces == ["gravity"] and second.forces == []


def test_simulation_config_default_backend_dependencies_and_snapshot(config_api: Any) -> None:
    values = dependencies()
    config = config_api.SimulationConfig(*values)
    assert config.backend == "Python"
    assert all(
        getattr(config, key) is value
        for key, value in zip(
            ("satellite", "environment", "state", "propagator", "solver"), values
        )
    )
    assert config.state_record is values[2].record
    assert config.julian_date == 2451545.0 and values[2].count == 2


@pytest.mark.parametrize("backend", ["Python", "C", "Rust", "JAX"])
def test_supported_simulation_backends(config_api: Any, backend: str) -> None:
    assert (
        config_api.SimulationConfig(*dependencies(), backend=backend).backend == backend
    )


def test_invalid_backend_fails_before_state_record_is_requested(config_api: Any) -> None:
    values = dependencies()
    with pytest.raises(ValueError, match="backend must be Python, C, Rust, or JAX"):
        config_api.SimulationConfig(*values, backend="invalid")
    assert values[2].count is None
