"""Contract tests for the Python satellite geometry dataclass."""

import numpy as np
import pytest

from GenADCS.backends.python.hardware.satellite_geometry import (
    SatelliteGeometry,
    SatellitePanel,
)


def test_geometry_requires_mass_and_inertia_and_preserves_supplied_values():
    inertia = np.diag([2.0, 3.0, 4.0])
    panels = [SatellitePanel((1, 0, 0), (1, 0, 0), (2, 1))]
    center = (0.1, -0.2, 0.3)
    geometry = SatelliteGeometry(12.5, inertia, panels, center)
    assert geometry.mass == 12.5
    assert geometry.inertia is inertia
    assert geometry.panels is panels
    assert geometry.center_of_mass is center
    np.testing.assert_array_equal(geometry.inertia, np.diag([2.0, 3.0, 4.0]))


def test_geometry_keyword_construction_and_defaults():
    inertia = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
    geometry = SatelliteGeometry(mass=4.0, inertia=inertia)
    assert geometry.mass == 4.0
    assert geometry.inertia is inertia
    assert geometry.panels == []
    assert geometry.center_of_mass == (0.0, 0.0, 0.0)


def test_default_panel_lists_are_independent_between_instances():
    first = SatelliteGeometry(1.0, np.eye(3))
    second = SatelliteGeometry(2.0, np.eye(3))
    assert first.panels is not second.panels
    first.panels.append(SatellitePanel((0, 0, 0), (1, 0, 0), (1, 1)))
    assert len(first.panels) == 1
    assert second.panels == []


def test_geometry_field_assignment_and_nested_panel_updates_are_live():
    panel = SatellitePanel((1, 0, 0), (1, 0, 0), (2, 1))
    geometry = SatelliteGeometry(5.0, np.eye(3), [panel])
    geometry.mass = 6.5
    geometry.center_of_mass = (0.25, 0.0, -0.1)
    geometry.panels[0].cd = 2.5
    assert geometry.mass == 6.5
    assert geometry.center_of_mass == (0.25, 0.0, -0.1)
    assert panel.cd == 2.5


def test_geometry_equality_repr_and_slots():
    inertia = np.diag([1.0, 2.0, 3.0])
    panel = SatellitePanel((0, 0, 0), (1, 0, 0), (1, 2))
    first = SatelliteGeometry(3.0, inertia, [panel])
    second = SatelliteGeometry(3.0, inertia, [panel])
    assert first == second
    assert "SatelliteGeometry" in repr(first)
    with pytest.raises(AttributeError):
        first.name = "test spacecraft"


def test_constructor_does_not_mutate_or_implicitly_copy_input_objects():
    inertia = np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]])
    panels = []
    geometry = SatelliteGeometry(7.0, inertia, panels)
    assert geometry.inertia is inertia
    assert geometry.panels is panels
    inertia[0, 0] = 9.0
    panels.append(SatellitePanel((0, 0, 0), (0, 1, 0), (2, 3)))
    assert geometry.inertia[0, 0] == 9.0
    assert len(geometry.panels) == 1
