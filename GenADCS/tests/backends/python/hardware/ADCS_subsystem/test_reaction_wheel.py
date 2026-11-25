"""Constructor, effects, and field tests for ReactionWheel."""

import pytest

from GenADCS.backends.python.bias import Bias
from GenADCS.backends.python.hardware.ADCS_subsystem import ReactionWheel
from GenADCS.backends.python.noise import Noise


class AddEffect:
    def __init__(self, amount, calls=None):
        self.amount = amount
        self.calls = calls

    def apply(self, value, dt, rng=None):
        if self.calls is not None:
            self.calls.append((value, dt, rng))
        return value + self.amount


def test_required_fields_and_defaults():
    wheel = ReactionWheel((1.0, 2.0, 3.0), (0.0, 0.0, 1.0), 0.04, 0.2, 5.0)
    assert wheel.position == (1.0, 2.0, 3.0)
    assert wheel.orientation == (0.0, 0.0, 1.0)
    assert wheel.rotational_inertia == 0.04
    assert wheel.max_torque == 0.2
    assert wheel.max_momentum == 5.0
    assert wheel.misalignment == (0.0, 0.0, 0.0)
    assert wheel.effects == []
    assert wheel.noises == []
    assert wheel.biases == []


def test_effect_default_lists_are_independent():
    first = ReactionWheel((0, 0, 0), (1, 0, 0), 0.03, 0.1, 2.0)
    second = ReactionWheel((0, 0, 0), (1, 0, 0), 0.03, 0.1, 2.0)
    assert first.effects is not second.effects
    first.effects.append(AddEffect(1))
    assert second.effects == []


def test_effects_reference_is_preserved_and_noise_bias_views_filter_in_order():
    noise_a, bias_a = Noise(), Bias()
    plain, noise_b, bias_b = object(), Noise(), Bias()
    effects = [noise_a, bias_a, plain, noise_b, bias_b]
    wheel = ReactionWheel((0, 0, 0), (0, 1, 0), 0.05, 0.15, 3.0, effects=effects)
    assert wheel.effects is effects
    assert wheel.noises == [noise_a, noise_b]
    assert wheel.biases == [bias_a, bias_b]
    effects.append(bias_a)
    assert wheel.biases == [bias_a, bias_b, bias_a]


def test_apply_errors_runs_effects_sequentially_and_passes_rng_and_dt():
    calls = []
    rng = object()
    wheel = ReactionWheel(
        (0, 0, 0),
        (1, 0, 0),
        0.04,
        0.2,
        4.0,
        effects=[AddEffect(-1.0, calls), AddEffect(0.25, calls)],
    )
    assert wheel.apply_errors(2.0, 0.5, rng) == 1.25
    assert calls == [(2.0, 0.5, rng), (1.0, 0.5, rng)]


def test_apply_errors_without_effects_returns_input_unchanged():
    value = [0.1, -0.2, 0.3]
    wheel = ReactionWheel((0, 0, 0), (0, 0, 1), 0.04, 0.2, 5.0)
    assert wheel.apply_errors(value, 1.0) is value


def test_fields_are_mutable_instance_is_slotted_and_missing_arguments_fail():
    wheel = ReactionWheel((0, 0, 0), (1, 0, 0), 0.03, 0.1, 2.0)
    wheel.position = (1, 0, 0)
    wheel.rotational_inertia = 0.05
    wheel.max_torque = 0.12
    wheel.max_momentum = 2.5
    wheel.misalignment = (0.01, 0.0, -0.01)
    assert wheel.position == (1, 0, 0)
    assert (wheel.rotational_inertia, wheel.max_torque, wheel.max_momentum) == (
        0.05,
        0.12,
        2.5,
    )
    assert wheel.misalignment == (0.01, 0.0, -0.01)
    with pytest.raises(AttributeError):
        wheel.extra = True
    with pytest.raises(TypeError):
        ReactionWheel((0, 0, 0), (1, 0, 0), 0.04, 0.2)


def test_constructor_preserves_supplied_mutable_values_without_coercion():
    position = [1, 2, 3]
    effects = []
    wheel = ReactionWheel(position, [0, 0, 1], 0.04, 0.2, 5.0, effects=effects)
    assert wheel.position is position
    assert wheel.effects is effects
    position[0] = 8
    assert wheel.position == [8, 2, 3]
