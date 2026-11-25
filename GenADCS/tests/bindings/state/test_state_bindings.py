"""Python-facing integration checks for the compiled state backends."""

import gc
from importlib import import_module
from types import ModuleType
from typing import Any

import numpy as np
import pytest


@pytest.fixture(params=("cpp", "rust"))
def backend(request: pytest.FixtureRequest) -> ModuleType:
    return import_module(f"GenADCS.backends.{request.param}.state")


def test_numpy_vectors_roundtrip_through_native_storage(backend: Any) -> None:
    orientation = np.array([2, 0, 0, 0], dtype=np.float32)
    rates = np.array([1, 2, 3], dtype=np.float32)
    state = backend.AttitudeState(orientation, rates)
    np.testing.assert_array_equal(state.quaternion_array, [1, 0, 0, 0])
    np.testing.assert_array_equal(state.rates, [1, 2, 3])
    assert state.rates.dtype == np.float64
    rates[0] = 99
    assert state.rates[0] == 1


def test_validation_exception_maps_to_python_value_error(backend: Any) -> None:
    with pytest.raises(ValueError):
        backend.AttitudeState([0, 0, 0, 0], [0, 0, 0])
    with pytest.raises(ValueError):
        backend.InternalState([np.nan])
    with pytest.raises(ValueError):
        backend.OrbitalState.from_cartesian([0, 0, 0], [0, 0, 0], np.inf)


def test_native_state_record_and_array_layout(backend: Any) -> None:
    state = backend.State(
        backend.AttitudeState([2, 0, 0, 0], [0.1, 0.2, 0.3]),
        backend.OrbitalState.from_cartesian([7e6, 0, 0], [0, 7500, 0], 2451545.0),
        backend.InternalState(np.array([0.1, -0.2], dtype=np.float32)),
    )
    record = state.to_record()
    assert record.dtype.names == (
        "quaternion",
        "rates",
        "position",
        "velocity",
        "momentum",
    )
    np.testing.assert_array_equal(record["quaternion"][0], [1, 0, 0, 0])
    np.testing.assert_allclose(record["momentum"][0], [0.1, -0.2], rtol=1e-6)
    assert (
        backend.State.from_record(record, 2451545.0).internal.wheel_momentum.size == 2
    )


def test_ephemeris_reference_survives_original_owner_destruction(backend: Any) -> None:
    class Ephemeris:
        def position(self, body: int, julian_date: float) -> np.ndarray:
            return np.array([body, julian_date, 0.0])

    ephemeris = Ephemeris()
    orbit = backend.OrbitalState.from_cartesian(
        [0, 0, 0], [0, 0, 0], 2451545.0, ephemeris
    )
    del ephemeris
    gc.collect()
    np.testing.assert_array_equal(orbit.sun_position, [0, 2451545.0, 0])
    cached = orbit.sun_position
    gc.collect()
    assert orbit.sun_position is cached
