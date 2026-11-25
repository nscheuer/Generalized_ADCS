"""Common constructor, defaults, and mutation contract for geometry panels."""

import gc
import weakref

import pytest


def test_required_fields_defaults_and_keyword_construction(geometry_api):
    panel = geometry_api.SatellitePanel(
        position=(1.0, -2.0, 3.5), orientation=(0.0, 0.6, 0.8), dimensions=(2.5, 1.25)
    )
    assert panel.position == (1.0, -2.0, 3.5)
    assert panel.orientation == (0.0, 0.6, 0.8)
    assert panel.dimensions == (2.5, 1.25)
    assert (panel.eta_s, panel.eta_d, panel.eta_a, panel.cd) == (0.0, 0.0, 1.0, 2.2)

    explicit = geometry_api.SatellitePanel(
        (0, 1, 0), (0, 0, 1), (3, 4), 0.15, 0.25, 0.60, 2.3
    )
    assert (explicit.eta_s, explicit.eta_d, explicit.eta_a, explicit.cd) == (
        0.15,
        0.25,
        0.60,
        2.3,
    )


def test_panel_fields_are_readable_and_mutable(geometry_api):
    panel = geometry_api.SatellitePanel((0, 0, 0), (1, 0, 0), (1, 2))
    panel.position = (1, 2, 3)
    panel.orientation = (0, 1, 0)
    panel.dimensions = (2, 3)
    panel.eta_s, panel.eta_d, panel.eta_a, panel.cd = 0.2, 0.3, 0.5, 2.0
    assert panel.position == (1, 2, 3)
    assert panel.orientation == (0, 1, 0)
    assert panel.dimensions == (2, 3)
    assert (panel.eta_s, panel.eta_d, panel.eta_a, panel.cd) == (0.2, 0.3, 0.5, 2.0)


def test_panel_keeps_python_objects_alive_and_releases_them(geometry_api):
    class Retained:
        pass

    value = Retained()
    reference = weakref.ref(value)
    panel = geometry_api.SatellitePanel(value, value, value)
    del value
    gc.collect()
    assert reference() is panel.position
    assert panel.position is panel.orientation is panel.dimensions
    del panel
    gc.collect()
    assert reference() is None


def test_missing_required_panel_arguments_raise_type_error(geometry_api):
    with pytest.raises(TypeError):
        geometry_api.SatellitePanel((0, 0, 0), (1, 0, 0))
