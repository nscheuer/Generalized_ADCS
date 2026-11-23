"""Common constructor, defaults, and nested ownership contract for geometry."""

import gc
import weakref

import numpy as np
import pytest


def test_geometry_required_fields_defaults_and_assignment(geometry_api):
    inertia = np.diag([2.0, 3.0, 4.0])
    geometry = geometry_api.SatelliteGeometry(mass=12.5, inertia=inertia)
    assert geometry.mass == 12.5
    np.testing.assert_array_equal(geometry.inertia, inertia)
    assert geometry.panels == []
    assert geometry.center_of_mass == (0.0, 0.0, 0.0)

    panel = geometry_api.SatellitePanel((1, 0, 0), (1, 0, 0), (2, 1))
    geometry.mass = 13.0
    geometry.center_of_mass = (0.1, -0.2, 0.3)
    geometry.panels = [panel]
    assert geometry.mass == 13.0
    assert geometry.center_of_mass == (0.1, -0.2, 0.3)
    assert len(geometry.panels) == 1
    assert geometry.panels[0] is panel


def test_each_geometry_gets_an_empty_panel_collection(geometry_api):
    first = geometry_api.SatelliteGeometry(1.0, np.eye(3))
    second = geometry_api.SatelliteGeometry(2.0, np.eye(3))
    first.panels = [geometry_api.SatellitePanel((0, 0, 0), (1, 0, 0), (1, 1))]
    assert len(first.panels) == 1
    assert second.panels == []


def test_geometry_keeps_constructor_python_objects_alive(geometry_api):
    class Retained:
        pass

    inertia = Retained()
    center = Retained()
    inertia_ref, center_ref = weakref.ref(inertia), weakref.ref(center)
    geometry = geometry_api.SatelliteGeometry(4.0, inertia, center_of_mass=center)
    del inertia, center
    gc.collect()
    assert inertia_ref() is geometry.inertia
    assert center_ref() is geometry.center_of_mass


def test_missing_required_geometry_arguments_raise_type_error(geometry_api):
    with pytest.raises(TypeError):
        geometry_api.SatelliteGeometry(inertia=np.eye(3))
