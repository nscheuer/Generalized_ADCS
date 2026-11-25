"""Tests for the Python ADCSSubsystem data container."""

import pytest

from GenADCS.backends.python.hardware.ADCS_subsystem import (
    ADCSSubsystem,
    Magnetorquer,
    ReactionWheel,
)


def test_default_actuator_collections_are_empty_and_independent():
    first, second = ADCSSubsystem(), ADCSSubsystem()
    assert first.reaction_wheels == []
    assert first.magnetorquers == []
    assert first.reaction_wheels is not second.reaction_wheels
    assert first.magnetorquers is not second.magnetorquers

    first.reaction_wheels.append(ReactionWheel((0, 0, 0), (1, 0, 0), 0.04, 0.2, 5.0))
    first.magnetorquers.append(Magnetorquer((0, 1, 0), (0, 0, 1), 0.3))
    assert len(first.reaction_wheels) == len(first.magnetorquers) == 1
    assert second.reaction_wheels == second.magnetorquers == []


def test_positional_and_keyword_construction_preserve_actuator_references():
    wheel = ReactionWheel((1, 2, 3), (0, 0, 1), 0.05, 0.15, 3.0)
    mtq = Magnetorquer((0, 1, 0), (1, 0, 0), 0.25)
    wheels, magnetorquers = [wheel], [mtq]
    positional = ADCSSubsystem(wheels, magnetorquers)
    keyword = ADCSSubsystem(reaction_wheels=wheels, magnetorquers=magnetorquers)
    assert positional.reaction_wheels is wheels
    assert positional.magnetorquers is magnetorquers
    assert keyword.reaction_wheels[0] is wheel
    assert keyword.magnetorquers[0] is mtq


def test_actuator_lists_are_live_mutable_fields():
    subsystem = ADCSSubsystem()
    wheel = ReactionWheel((0, 0, 0), (0, 1, 0), 0.03, 0.1, 2.0)
    mtq = Magnetorquer((0, 0, 1), (1, 0, 0), 0.2)
    subsystem.reaction_wheels = [wheel]
    subsystem.magnetorquers = [mtq]
    subsystem.reaction_wheels[0].max_torque = 0.12
    assert subsystem.reaction_wheels[0] is wheel
    assert wheel.max_torque == 0.12
    assert subsystem.magnetorquers[0] is mtq


def test_value_equality_repr_and_slotted_instance_contract():
    wheel_a = ReactionWheel((0, 0, 0), (0, 0, 1), 0.04, 0.2, 5.0)
    wheel_b = ReactionWheel((0, 0, 0), (0, 0, 1), 0.04, 0.2, 5.0)
    first = ADCSSubsystem([wheel_a], [])
    equivalent = ADCSSubsystem([wheel_b], [])
    assert first == equivalent
    assert "ADCSSubsystem" in repr(first)
    with pytest.raises(AttributeError):
        first.description = "extra field"


def test_constructor_does_not_validate_or_copy_user_supplied_lists():
    wheels = [object()]
    magnetorquers = []
    subsystem = ADCSSubsystem(wheels, magnetorquers)
    assert subsystem.reaction_wheels is wheels
    assert subsystem.magnetorquers is magnetorquers
    wheels.append("caller mutation")
    assert subsystem.reaction_wheels[-1] == "caller mutation"
