"""Contract tests for the Python satellite panel dataclass."""

import pytest

from GenADCS.backends.python.hardware.satellite_geometry import SatellitePanel


def test_panel_constructor_preserves_all_explicit_values():
    panel = SatellitePanel(
        position=(1.0, -2.0, 3.5),
        orientation=(0.0, 0.6, 0.8),
        dimensions=(2.5, 1.25),
        eta_s=0.2,
        eta_d=0.3,
        eta_a=0.5,
        cd=2.1,
    )
    assert panel.position == (1.0, -2.0, 3.5)
    assert panel.orientation == (0.0, 0.6, 0.8)
    assert panel.dimensions == (2.5, 1.25)
    assert (panel.eta_s, panel.eta_d, panel.eta_a, panel.cd) == (0.2, 0.3, 0.5, 2.1)


def test_panel_defaults_follow_dataclass_contract():
    panel = SatellitePanel((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 2.0))
    assert panel.eta_s == 0.0
    assert panel.eta_d == 0.0
    assert panel.eta_a == 1.0
    assert panel.cd == 2.2


def test_panel_supports_positional_and_keyword_construction_with_value_equality():
    positional = SatellitePanel((1, 2, 3), (0, 0, 1), (4, 5), 0.1, 0.2, 0.7, 2.0)
    keyword = SatellitePanel(
        dimensions=(4, 5), orientation=(0, 0, 1), position=(1, 2, 3),
        eta_s=0.1, eta_d=0.2, eta_a=0.7, cd=2.0,
    )
    assert positional == keyword
    assert "SatellitePanel" in repr(positional)
    assert "dimensions=(4, 5)" in repr(positional)


def test_panel_fields_are_mutable_but_instances_are_slotted():
    panel = SatellitePanel((0, 0, 0), (1, 0, 0), (1, 1))
    panel.cd = 2.4
    panel.position = (1, 0, 0)
    assert panel.cd == 2.4
    assert panel.position == (1, 0, 0)
    with pytest.raises(AttributeError):
        panel.extra_metadata = "not a declared dataclass field"


def test_panel_retains_supplied_container_references_without_coercion():
    position = [1.0, 2.0, 3.0]
    dimensions = [4.0, 5.0]
    panel = SatellitePanel(position, (0.0, 0.0, 1.0), dimensions)
    assert panel.position is position
    assert panel.dimensions is dimensions
    position[0] = 9.0
    dimensions.append(6.0)
    assert panel.position == [9.0, 2.0, 3.0]
    assert panel.dimensions == [4.0, 5.0, 6.0]
