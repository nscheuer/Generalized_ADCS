"""Python-facing ownership, constructor, and behavior tests for ADCS bindings."""

import gc
import weakref

import pytest


def make_wheel(api):
    return api.ReactionWheel((1, 0, 0), (0, 0, 1), 0.02, 0.1, 2.0)


def make_magnetorquer(api):
    return api.Magnetorquer((0, 1, 0), (1, 0, 0), 0.25)


def test_subsystem_defaults_and_list_ownership_quirks(
    subsystem_backend, subsystem_backend_name
):
    first, second = subsystem_backend.ADCSSubsystem(), subsystem_backend.ADCSSubsystem()
    assert first.reaction_wheels == first.magnetorquers == []
    first.reaction_wheels = [make_wheel(subsystem_backend)]
    assert second.reaction_wheels == []

    wheel, torquer = make_wheel(subsystem_backend), make_magnetorquer(subsystem_backend)
    wheels, torquers = [wheel], [torquer]
    system = subsystem_backend.ADCSSubsystem(wheels, torquers)
    assert system.reaction_wheels[0] is wheel and system.magnetorquers[0] is torquer
    if subsystem_backend_name == "cpp":
        assert system.reaction_wheels is wheels
        wheels.append(make_wheel(subsystem_backend))
        assert len(system.reaction_wheels) == 2
    else:
        assert system.reaction_wheels is not wheels
        wheels.append(make_wheel(subsystem_backend))
        assert len(system.reaction_wheels) == 1
        exposed = system.reaction_wheels
        exposed.clear()
        assert len(system.reaction_wheels) == 1
        system.reaction_wheels = exposed
        assert system.reaction_wheels == []


def test_device_and_subsystem_keep_references_and_release_on_replacement(
    subsystem_backend,
):
    class Retained:
        pass

    position = Retained()
    position_ref = weakref.ref(position)
    wheel = subsystem_backend.ReactionWheel(position, position, 0.1, 0.2, 1.0)
    del position
    gc.collect()
    assert wheel.position is wheel.orientation is position_ref()

    old_effect = Retained()
    old_ref = weakref.ref(old_effect)
    wheel.effects = [old_effect]
    wheel.effects = []
    del old_effect
    gc.collect()
    assert old_ref() is None

    retained_effect = Retained()
    retained_effect_ref = weakref.ref(retained_effect)
    wheel.effects = [retained_effect]
    system = subsystem_backend.ADCSSubsystem([wheel], [])
    del wheel, retained_effect
    gc.collect()
    assert system.reaction_wheels[0].effects[0] is retained_effect_ref()
    del system
    gc.collect()
    assert retained_effect_ref() is None
