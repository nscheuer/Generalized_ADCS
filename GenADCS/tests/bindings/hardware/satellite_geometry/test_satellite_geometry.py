"""Python-facing lifetime and data-container tests for geometry bindings."""

import gc
import weakref

import numpy as np
import pytest


def make_panel(api, **overrides):
    values = dict(
        position=(1.0, 2.0, 3.0), orientation=(0.0, 0.0, 1.0), dimensions=(2.0, 1.0)
    )
    values.update(overrides)
    return api.SatellitePanel(**values)


def test_geometry_constructor_defaults_and_mutable_fields(geometry_backend):
    inertia = np.diag([1.0, 2.0, 3.0])
    geometry = geometry_backend.SatelliteGeometry(5.0, inertia)
    assert geometry.mass == 5.0
    assert geometry.inertia is inertia
    assert geometry.panels == []
    assert geometry.center_of_mass == (0.0, 0.0, 0.0)

    geometry.mass = 6.5
    geometry.inertia = np.eye(3)
    geometry.center_of_mass = (0.25, -0.1, 0.0)
    assert geometry.mass == 6.5
    np.testing.assert_array_equal(geometry.inertia, np.eye(3))
    assert geometry.center_of_mass == (0.25, -0.1, 0.0)


def test_geometry_constructor_retains_panel_objects_and_panel_mutability(
    geometry_backend,
):
    panel = make_panel(geometry_backend)
    geometry = geometry_backend.SatelliteGeometry(3.0, np.eye(3), [panel])
    assert geometry.panels[0] is panel
    geometry.panels = [panel]
    geometry.panels[0].cd = 2.6
    assert panel.cd == 2.6


def test_geometry_center_and_inertia_references_survive_gc(geometry_backend):
    class RetainedValue:
        pass

    inertia = RetainedValue()
    center = RetainedValue()
    inertia_ref, center_ref = weakref.ref(inertia), weakref.ref(center)
    geometry = geometry_backend.SatelliteGeometry(1.0, inertia, center_of_mass=center)
    del inertia, center
    gc.collect()
    assert inertia_ref() is geometry.inertia
    assert center_ref() is geometry.center_of_mass


def test_default_panel_lists_are_independent_and_can_be_reassigned(geometry_backend):
    first = geometry_backend.SatelliteGeometry(1.0, np.eye(3))
    second = geometry_backend.SatelliteGeometry(2.0, np.eye(3))
    first.panels = [make_panel(geometry_backend)]
    assert len(first.panels) == 1
    assert second.panels == []


def test_panel_list_constructor_and_getter_ownership_quirks(
    geometry_backend, geometry_backend_name
):
    panel = make_panel(geometry_backend)
    source = [panel]
    geometry = geometry_backend.SatelliteGeometry(2.0, np.eye(3), source)

    if geometry_backend_name == "cpp":
        # pybind stores the original Python list object.
        assert geometry.panels is source
        source.append(make_panel(geometry_backend))
        assert len(geometry.panels) == 2
    else:
        # PyO3 extracts Vec<PyObject> and creates a new list for each getter.
        assert geometry.panels is not source
        source.append(make_panel(geometry_backend))
        assert len(geometry.panels) == 1
        retrieved = geometry.panels
        retrieved.append(make_panel(geometry_backend))
        assert len(geometry.panels) == 1
        geometry.panels = retrieved
        assert len(geometry.panels) == 2


def test_geometry_panel_property_assignment_sequence_support(
    geometry_backend, geometry_backend_name
):
    geometry = geometry_backend.SatelliteGeometry(2.0, np.eye(3))
    panel = make_panel(geometry_backend)
    geometry.panels = [panel]
    assert len(geometry.panels) == 1
    assert geometry.panels[0] is panel
    if geometry_backend_name == "cpp":
        with pytest.raises(TypeError):
            geometry.panels = (panel,)
    else:
        geometry.panels = (panel,)
        assert len(geometry.panels) == 1


def test_geometry_constructor_arguments_are_checked_by_binding(geometry_backend):
    with pytest.raises(TypeError):
        geometry_backend.SatelliteGeometry(inertia=np.eye(3))
    with pytest.raises(TypeError):
        geometry_backend.SatelliteGeometry("not a mass", np.eye(3))


def test_geometry_equality_behavior_is_explicit_for_each_backend(
    geometry_backend, geometry_backend_name
):
    geometry_a = geometry_backend.SatelliteGeometry(1.0, np.eye(3))
    geometry_b = geometry_backend.SatelliteGeometry(1.0, np.eye(3))

    if geometry_backend_name == "cpp":
        assert geometry_a == geometry_b
    else:
        # These PyO3 classes currently use Python's default identity equality.
        assert geometry_a != geometry_b
    assert geometry_a == geometry_a
