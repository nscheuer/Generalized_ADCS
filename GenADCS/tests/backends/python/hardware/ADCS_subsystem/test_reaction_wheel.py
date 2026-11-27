"""Constructor, noise/bias, and field tests for ReactionWheel."""

import pytest

from GenADCS.backends.python.hardware.ADCS_subsystem import ReactionWheel


def test_required_fields_and_defaults() -> None:
    wheel = ReactionWheel((1.0, 2.0, 3.0), (0.0, 0.0, 1.0), 0.04, 0.2, 5.0)
    assert wheel.position == (1.0, 2.0, 3.0)
    assert wheel.orientation == (0.0, 0.0, 1.0)
    assert wheel.rotational_inertia == 0.04
    assert wheel.max_torque == 0.2
    assert wheel.max_momentum == 5.0
    assert wheel.misalignment == (0.0, 0.0, 0.0)
    assert wheel.noise == 0.0
    assert wheel.bias == 0.0


def test_noise_and_bias_validation() -> None:
    wheel = ReactionWheel((0, 0, 0), (0, 1, 0), 0.05, 0.15, 3.0, noise=0.01, bias=-0.02)
    assert wheel.noise == 0.01
    assert wheel.bias == -0.02
    with pytest.raises(ValueError):
        ReactionWheel((0, 0, 0), (0, 1, 0), 0.05, 0.15, 3.0, noise=-0.1)
    with pytest.raises(ValueError):
        ReactionWheel((0, 0, 0), (0, 1, 0), 0.05, 0.15, 3.0, bias=float("nan"))


def test_fields_are_mutable_instance_is_slotted_and_missing_arguments_fail() -> None:
    wheel = ReactionWheel((0, 0, 0), (1, 0, 0), 0.03, 0.1, 2.0)
    wheel.position = (1, 0, 0)
    wheel.rotational_inertia = 0.05
    wheel.max_torque = 0.12
    wheel.max_momentum = 2.5
    wheel.misalignment = (0.01, 0.0, -0.01)
    wheel.noise = 0.03
    wheel.bias = 0.04
    assert wheel.position == (1, 0, 0)
    assert (wheel.rotational_inertia, wheel.max_torque, wheel.max_momentum) == (
        0.05,
        0.12,
        2.5,
    )
    assert wheel.misalignment == (0.01, 0.0, -0.01)
    assert (wheel.noise, wheel.bias) == (0.03, 0.04)
    with pytest.raises(AttributeError):
        wheel.extra = True
    with pytest.raises(TypeError):
        ReactionWheel((0, 0, 0), (1, 0, 0), 0.04, 0.2)


def test_constructor_preserves_supplied_mutable_values_without_coercion() -> None:
    position = [1, 2, 3]
    wheel = ReactionWheel(position, [0, 0, 1], 0.04, 0.2, 5.0)
    assert wheel.position is position
    position[0] = 8
    assert wheel.position == [8, 2, 3]
