"""Python-facing lifetime and data-container tests for panel bindings."""

import gc
import weakref
from typing import Any

import pytest


def test_panel_constructor_defaults_and_explicit_fields(geometry_backend: Any) -> None:
    panel = geometry_backend.SatellitePanel(
        (1.0, 2.0, 3.0), (0.0, 0.0, 1.0), (2.0, 1.0)
    )
    assert panel.position == (1.0, 2.0, 3.0)
    assert panel.orientation == (0.0, 0.0, 1.0)
    assert panel.dimensions == (2.0, 1.0)
    assert (panel.eta_s, panel.eta_d, panel.eta_a, panel.cd) == (0.0, 0.0, 1.0, 2.2)

    explicit = geometry_backend.SatellitePanel(
        [1, -2, 3], [0.0, 0.6, 0.8], [4, 5], eta_s=0.2, eta_d=0.3, eta_a=0.5, cd=2.1
    )
    assert explicit.position == [1, -2, 3]
    assert explicit.orientation == [0.0, 0.6, 0.8]
    assert explicit.dimensions == [4, 5]
    assert (explicit.eta_s, explicit.eta_d, explicit.eta_a, explicit.cd) == (
        0.2,
        0.3,
        0.5,
        2.1,
    )


def test_panel_fields_can_be_read_and_mutated(geometry_backend: Any) -> None:
    panel = geometry_backend.SatellitePanel(
        (1.0, 2.0, 3.0), (0.0, 0.0, 1.0), (2.0, 1.0)
    )
    panel.position = (9.0, 8.0, 7.0)
    panel.orientation = (1.0, 0.0, 0.0)
    panel.dimensions = (3.0, 4.0)
    panel.eta_s, panel.eta_d, panel.eta_a, panel.cd = 0.1, 0.2, 0.7, 2.4
    assert panel.position == (9.0, 8.0, 7.0)
    assert panel.orientation == (1.0, 0.0, 0.0)
    assert panel.dimensions == (3.0, 4.0)
    assert (panel.eta_s, panel.eta_d, panel.eta_a, panel.cd) == (0.1, 0.2, 0.7, 2.4)


def test_panel_retains_python_objects_until_replaced_or_destroyed(geometry_backend: Any) -> None:
    class CoordinateData:
        pass

    coordinates = CoordinateData()
    coordinates_ref = weakref.ref(coordinates)
    panel = geometry_backend.SatellitePanel(coordinates, coordinates, coordinates)
    del coordinates
    gc.collect()
    assert coordinates_ref() is not None
    assert panel.position is panel.orientation is panel.dimensions

    replacement = CoordinateData()
    replacement_ref = weakref.ref(replacement)
    panel.position = replacement
    del replacement
    gc.collect()
    assert replacement_ref() is panel.position
    del panel
    gc.collect()
    assert coordinates_ref() is None


def test_panel_constructor_arguments_are_checked_by_binding(geometry_backend: Any) -> None:
    with pytest.raises(TypeError):
        geometry_backend.SatellitePanel((0, 0, 0), (1, 0, 0))


def test_panel_equality_behavior_is_explicit_for_each_backend(
    geometry_backend: Any, geometry_backend_name: str
) -> None:
    panel_a = geometry_backend.SatellitePanel((1, 2, 3), (0, 0, 1), (4, 5))
    panel_b = geometry_backend.SatellitePanel((1, 2, 3), (0, 0, 1), (4, 5))
    if geometry_backend_name == "cpp":
        assert panel_a == panel_b
    else:
        # These PyO3 classes currently use Python's default identity equality.
        assert panel_a != panel_b
