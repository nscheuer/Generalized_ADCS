"""Contract tests for DynamicsConfig's force and torque collections."""

import pytest

from GenADCS.backends.python.config import DynamicsConfig


def test_defaults_create_independent_empty_collections():
    first = DynamicsConfig()
    second = DynamicsConfig()

    assert first.forces == []
    assert first.torques == []
    assert first.forces is not second.forces
    assert first.torques is not second.torques

    first.forces.append("drag")
    first.torques.append("magnetic")
    assert second.forces == []
    assert second.torques == []


def test_positional_and_keyword_construction_preserve_collection_references():
    forces = [object(), object()]
    torques = [object()]

    positional = DynamicsConfig(forces, torques)
    keyword = DynamicsConfig(torques=torques, forces=forces)

    assert positional.forces is forces
    assert positional.torques is torques
    assert keyword.forces is forces
    assert keyword.torques is torques
    assert positional == keyword


def test_collections_and_fields_are_mutable():
    force = object()
    torque = object()
    config = DynamicsConfig()

    config.forces.append(force)
    config.torques.append(torque)
    assert config.forces == [force]
    assert config.torques == [torque]

    replacement_forces = ["gravity"]
    replacement_torques = ["control"]
    config.forces = replacement_forces
    config.torques = replacement_torques
    assert config.forces is replacement_forces
    assert config.torques is replacement_torques


def test_repr_equality_and_slots_follow_dataclass_contract():
    first = DynamicsConfig(forces=["drag"], torques=["gravity-gradient"])
    equal = DynamicsConfig(forces=["drag"], torques=["gravity-gradient"])
    different = DynamicsConfig(forces=["drag"], torques=[])

    assert first == equal
    assert first != different
    assert "DynamicsConfig" in repr(first)
    assert "forces=['drag']" in repr(first)
    with pytest.raises(AttributeError):
        first.extra = "not a declared field"


@pytest.mark.parametrize("forces, torques", [(None, []), ([], None), (None, None)])
def test_config_does_not_impose_collection_type_validation(forces, torques):
    """The dataclass stores supplied values without coercing or validating them."""
    config = DynamicsConfig(forces=forces, torques=torques)
    assert config.forces is forces
    assert config.torques is torques
