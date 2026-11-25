"""Field parity for SatelliteGeometry across backends."""

from typing import Any

import numpy as np


def test_geometry_defaults_and_explicit_fields_match_pairwise(geometry_backends: Any) -> None:
    defaults = {
        name: api.SatelliteGeometry(2.0, np.eye(3))
        for name, api in geometry_backends.items()
    }
    for backend, geometry in defaults.items():
        assert geometry.mass == 2.0, backend
        np.testing.assert_array_equal(geometry.inertia, np.eye(3), err_msg=backend)
        assert geometry.panels == [], backend
        assert geometry.center_of_mass == (0.0, 0.0, 0.0), backend

    panel_results = {
        name: api.SatellitePanel((1, 2, 3), (0, 0, 1), (4, 5), 0.1, 0.2, 0.7, 2.0)
        for name, api in geometry_backends.items()
    }
    geometries = {
        name: api.SatelliteGeometry(
            7.5, np.diag([2.0, 3.0, 4.0]), [panel_results[name]], (0.25, -0.1, 0.0)
        )
        for name, api in geometry_backends.items()
    }
    reference = geometries["python"]
    for backend, geometry in geometries.items():
        assert geometry.mass == reference.mass, backend
        np.testing.assert_array_equal(
            geometry.inertia, reference.inertia, err_msg=backend
        )
        assert geometry.center_of_mass == reference.center_of_mass, backend
        assert len(geometry.panels) == len(reference.panels) == 1, backend
        assert geometry.panels[0].position == reference.panels[0].position, backend
        assert geometry.panels[0].cd == reference.panels[0].cd, backend


def test_geometry_mutable_fields_match_pairwise(geometry_backends: Any) -> None:
    results = {}
    for name, api in geometry_backends.items():
        geometry = api.SatelliteGeometry(1.0, np.eye(3))
        geometry.mass = 3.5
        geometry.inertia = np.diag([4.0, 5.0, 6.0])
        geometry.center_of_mass = (0.1, 0.2, 0.3)
        geometry.panels = [api.SatellitePanel((0, 1, 0), (0, 0, 1), (3, 2))]
        results[name] = geometry
    reference = results["python"]
    for backend, geometry in results.items():
        assert geometry.mass == reference.mass, backend
        np.testing.assert_array_equal(
            geometry.inertia, reference.inertia, err_msg=backend
        )
        assert geometry.center_of_mass == reference.center_of_mass, backend
        assert geometry.panels[0].dimensions == reference.panels[0].dimensions, backend
