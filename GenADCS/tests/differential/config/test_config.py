from types import SimpleNamespace
from typing import Any


class State:
    def __init__(self, date: Any) -> None:
        self.orbit = SimpleNamespace(julian_date=date)
        self.record = object()
        self.calls: list[Any] = []

    def to_record(self, wheel_count: Any) -> Any:
        self.calls.append(wheel_count)
        return self.record


def test_dynamics_values_match_across_backends(config_backends: Any) -> None:
    configs: dict[str, Any] = {
        name: api.DynamicsConfig(["drag"], ["magnetic"])
        for name, api in config_backends.items()
    }
    for name, config in configs.items():
        assert (config.forces, config.torques) == (["drag"], ["magnetic"]), name


def test_simulation_derived_values_and_dependencies_match(config_backends: Any) -> None:
    configs: dict[str, tuple[Any, tuple[Any, ...], State]] = {}
    for name, api in config_backends.items():
        sat = SimpleNamespace(adcs=SimpleNamespace(reaction_wheels=[None, None, None]))
        state = State(2460000.25)
        deps = (sat, object(), state, object(), object())
        configs[name] = (api.SimulationConfig(*deps, backend="Rust"), deps, state)
    for name, (config, deps, state) in configs.items():
        assert config.backend == "Rust", name
        assert config.state_record is state.record, name
        assert config.julian_date == 2460000.25 and state.calls == [3], name
        assert all(
            getattr(config, key) is value
            for key, value in zip(
                ("satellite", "environment", "state", "propagator", "solver"), deps
            )
        ), name
