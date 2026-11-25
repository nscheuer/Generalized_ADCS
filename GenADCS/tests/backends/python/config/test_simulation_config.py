"""Contract tests for SimulationConfig initialization and state snapshots."""

from types import SimpleNamespace

import numpy as np
import pytest

from GenADCS.backends.python.config import SimulationConfig


class StateRecordSource:
    """Small state double exposing exactly what SimulationConfig consumes."""

    def __init__(self, julian_date=2451545.0, record=None):
        self.orbit = SimpleNamespace(julian_date=julian_date)
        self.record = (
            np.array([(1.0,)], dtype=[("sentinel", np.float64)])
            if record is None
            else record
        )
        self.wheel_counts = []

    def to_record(self, wheel_count):
        self.wheel_counts.append(wheel_count)
        return self.record


def make_dependencies(*, wheel_count=0, julian_date=2451545.0, record=None):
    satellite = SimpleNamespace(
        adcs=SimpleNamespace(reaction_wheels=[object() for _ in range(wheel_count)])
    )
    state = StateRecordSource(julian_date=julian_date, record=record)
    return satellite, SimpleNamespace(), state, SimpleNamespace(), SimpleNamespace()


def test_default_backend_and_required_dependencies_are_preserved():
    dependencies = make_dependencies(wheel_count=2)
    config = SimulationConfig(*dependencies)

    assert config.satellite is dependencies[0]
    assert config.environment is dependencies[1]
    assert config.state is dependencies[2]
    assert config.propagator is dependencies[3]
    assert config.solver is dependencies[4]
    assert config.backend == "Python"
    assert dependencies[2].wheel_counts == [2]


@pytest.mark.parametrize("backend", ["Python", "C", "Rust", "JAX"])
def test_each_supported_backend_is_accepted(backend):
    dependencies = make_dependencies()
    config = SimulationConfig(*dependencies, backend=backend)
    assert config.backend == backend


@pytest.mark.parametrize("backend", ["", "python", "c", "CUDA", None, 1])
def test_unsupported_backend_is_rejected_with_clear_error(backend):
    dependencies = make_dependencies()
    with pytest.raises(ValueError, match="backend must be Python, C, Rust, or JAX"):
        SimulationConfig(*dependencies, backend=backend)
    # Backend validation happens before the state is consulted.
    assert dependencies[2].wheel_counts == []


def test_state_record_and_julian_date_are_initialized_from_dependencies():
    record = np.zeros(
        1, dtype=[("position", np.float64, (3,)), ("momentum", np.float64, (2,))]
    )
    record["position"][0] = [7.0e6, 1.0, -2.0]
    record["momentum"][0] = [0.25, -0.5]
    dependencies = make_dependencies(
        wheel_count=2, julian_date=2460000.25, record=record
    )

    config = SimulationConfig(*dependencies)

    assert config.state_record is record
    np.testing.assert_array_equal(config.state_record["position"], [[7.0e6, 1.0, -2.0]])
    np.testing.assert_array_equal(config.state_record["momentum"], [[0.25, -0.5]])
    assert config.julian_date == 2460000.25
    assert dependencies[2].wheel_counts == [2]


def test_snapshot_fields_are_not_constructor_arguments_but_remain_mutable():
    dependencies = make_dependencies()
    with pytest.raises(TypeError):
        SimulationConfig(*dependencies, state_record=np.empty(1), julian_date=2451545.0)

    config = SimulationConfig(*dependencies)
    new_record = np.zeros(2)
    config.state_record = new_record
    config.julian_date = 2452000.0
    assert config.state_record is new_record
    assert config.julian_date == 2452000.0


def test_config_uses_the_wheel_count_when_building_the_state_record():
    dependencies = make_dependencies(wheel_count=3)
    config = SimulationConfig(*dependencies)

    assert dependencies[2].wheel_counts == [3]
    assert config.state_record is dependencies[2].record


def test_slots_and_repr_hide_internal_state_record():
    config = SimulationConfig(*make_dependencies())
    representation = repr(config)

    assert "SimulationConfig" in representation
    assert "backend='Python'" in representation
    assert "state_record" not in representation
    with pytest.raises(AttributeError):
        config.extra = "not a declared field"


def test_julian_date_is_copied_as_a_value_at_initialization():
    dependencies = make_dependencies(julian_date=2451545.0)
    config = SimulationConfig(*dependencies)

    dependencies[2].orbit.julian_date = 2460000.0
    assert config.julian_date == 2451545.0
