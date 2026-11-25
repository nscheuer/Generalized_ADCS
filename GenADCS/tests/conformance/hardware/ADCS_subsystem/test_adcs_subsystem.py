from typing import Any


def test_default_lists_are_independent(adcs_api: Any) -> None:
    first, second = adcs_api.ADCSSubsystem(), adcs_api.ADCSSubsystem()
    assert first.reaction_wheels == second.reaction_wheels == []
    assert first.magnetorquers == second.magnetorquers == []
    first.reaction_wheels = [object()]
    assert second.reaction_wheels == []


def test_constructor_and_assignment_keep_device_items(adcs_api: Any) -> None:
    wheel = adcs_api.ReactionWheel((0, 0, 0), (0, 0, 1), 0.1, 0.2, 1.0)
    torquer = adcs_api.Magnetorquer((0, 0, 0), (0, 0, 1), 0.3)
    system = adcs_api.ADCSSubsystem([wheel], [torquer])
    assert system.reaction_wheels[0] is wheel
    assert system.magnetorquers[0] is torquer
    system.magnetorquers = [torquer]
    assert len(system.magnetorquers) == 1
    assert system.magnetorquers[0] is torquer
