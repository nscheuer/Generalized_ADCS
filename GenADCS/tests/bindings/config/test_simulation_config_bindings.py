"""Cross-backend tests for SimulationConfig's Python binding contract."""

import gc
import weakref
from types import SimpleNamespace

import pytest


class StateDouble:
    def __init__(self, date=2451545.0):
        self.orbit = SimpleNamespace(julian_date=date)
        self.record = object()
        self.wheel_counts = []

    def to_record(self, wheel_count):
        self.wheel_counts.append(wheel_count)
        return self.record


def make_dependencies(wheels=2, date=2451545.0):
    satellite = SimpleNamespace(
        adcs=SimpleNamespace(reaction_wheels=[object() for _ in range(wheels)])
    )
    state = StateDouble(date)
    return satellite, SimpleNamespace(), state, SimpleNamespace(), SimpleNamespace()


def test_constructor_defaults_backend_and_initializes_state_snapshots(config_backend):
    values = make_dependencies(wheels=2, date=2460000.25)
    config = config_backend.SimulationConfig(*values)

    for name, expected in zip(
        ("satellite", "environment", "state", "propagator", "solver"), values
    ):
        assert getattr(config, name) is expected
    assert config.backend == "Python"
    assert config.state_record is values[2].record
    assert config.julian_date == 2460000.25
    assert values[2].wheel_counts == [2]


@pytest.mark.parametrize("backend", ["Python", "C", "Rust", "JAX"])
def test_supported_backend_names_are_accepted(config_backend, backend):
    assert (
        config_backend.SimulationConfig(*make_dependencies(), backend=backend).backend
        == backend
    )


@pytest.mark.parametrize("backend", ["", "python", "CUDA", None])
def test_unsupported_backend_names_are_rejected(config_backend, backend):
    values = make_dependencies()
    expected_error = (
        TypeError
        if config_backend.__name__.endswith("rust.config") and backend is None
        else ValueError
    )
    with pytest.raises(expected_error):
        config_backend.SimulationConfig(*values, backend=backend)
    assert values[2].wheel_counts == []


def test_missing_required_constructor_arguments_are_rejected(config_backend):
    with pytest.raises(TypeError):
        config_backend.SimulationConfig()


def test_snapshot_property_mutability_is_backend_specific(
    config_backend, config_backend_name
):
    config = config_backend.SimulationConfig(*make_dependencies())
    replacement = object()
    if config_backend_name == "python":
        config.state_record = replacement
        config.julian_date = 2452000.0
        assert config.state_record is replacement
        assert config.julian_date == 2452000.0
    else:
        # Native bindings expose these derived values as getter-only properties.
        with pytest.raises(AttributeError):
            config.state_record = replacement
        with pytest.raises(AttributeError):
            config.julian_date = 2452000.0


def test_dependency_and_backend_properties_remain_mutable_after_construction(
    config_backend,
):
    config = config_backend.SimulationConfig(*make_dependencies())
    new_solver = object()
    config.solver = new_solver
    assert config.solver is new_solver
    config.backend = "not-validated-on-assignment"
    assert config.backend == "not-validated-on-assignment"


def test_native_backend_requires_string_backend_value(
    config_backend, config_backend_name
):
    if config_backend_name == "python":
        with pytest.raises(ValueError):
            config_backend.SimulationConfig(*make_dependencies(), backend=1)
    elif config_backend_name == "rust":
        with pytest.raises(TypeError):
            config_backend.SimulationConfig(*make_dependencies(), backend=1)
    else:
        # pybind converts the object to text before checking allowed names.
        with pytest.raises(ValueError):
            config_backend.SimulationConfig(*make_dependencies(), backend=1)


def test_constructor_retains_python_dependencies_until_config_is_destroyed(
    config_backend,
):
    class Dependency:
        pass

    satellite = Dependency()
    satellite.adcs = SimpleNamespace(reaction_wheels=[object()])
    environment, propagator, solver = Dependency(), Dependency(), Dependency()
    state = StateDouble()
    state.record = Dependency()
    dependencies = (satellite, environment, state, propagator, solver)
    dependency_refs = tuple(weakref.ref(item) for item in dependencies)
    record_ref = weakref.ref(state.record)

    config = config_backend.SimulationConfig(*dependencies)
    del dependencies, satellite, environment, state, propagator, solver
    gc.collect()
    assert all(reference() is not None for reference in dependency_refs)
    assert record_ref() is not None

    # Overwriting the dependency properties releases their stored references;
    # the derived state record remains independently owned by the config.
    config.satellite = Dependency()
    config.environment = Dependency()
    config.state = Dependency()
    config.propagator = Dependency()
    config.solver = Dependency()
    gc.collect()
    assert all(reference() is None for reference in dependency_refs)
    assert config.state_record is record_ref()

    del config
    gc.collect()
    assert record_ref() is None


def test_constructor_errors_from_dependency_access_are_not_silenced(config_backend):
    values = make_dependencies()
    del values[2].orbit
    with pytest.raises(AttributeError, match="orbit"):
        config_backend.SimulationConfig(*values)
