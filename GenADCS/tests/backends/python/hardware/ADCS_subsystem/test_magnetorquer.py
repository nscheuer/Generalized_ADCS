"""Constructor, effects, and field tests for Magnetorquer."""

import pytest

from GenADCS.backends.python.bias import Bias
from GenADCS.backends.python.hardware.ADCS_subsystem import Magnetorquer
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
    mtq = Magnetorquer((1.0, 2.0, 3.0), (0.0, 0.0, 1.0), 0.25)
    assert mtq.position == (1.0, 2.0, 3.0)
    assert mtq.orientation == (0.0, 0.0, 1.0)
    assert mtq.max_dipole == 0.25
    assert mtq.misalignment == (0.0, 0.0, 0.0)
    assert mtq.effects == []
    assert mtq.noises == []
    assert mtq.biases == []


def test_effect_default_lists_are_independent():
    first = Magnetorquer((0, 0, 0), (1, 0, 0), 0.1)
    second = Magnetorquer((0, 0, 0), (1, 0, 0), 0.1)
    assert first.effects is not second.effects
    first.effects.append(AddEffect(1))
    assert second.effects == []


def test_effects_reference_is_preserved_and_noise_bias_views_filter_in_order():
    noise_a, bias_a = Noise(), Bias()
    plain, noise_b, bias_b = object(), Noise(), Bias()
    effects = [noise_a, bias_a, plain, noise_b, bias_b]
    mtq = Magnetorquer((0, 0, 0), (0, 1, 0), 0.3, effects=effects)
    assert mtq.effects is effects
    assert mtq.noises == [noise_a, noise_b]
    assert mtq.biases == [bias_a, bias_b]
    effects.append(noise_a)
    assert mtq.noises == [noise_a, noise_b, noise_a]


def test_apply_errors_runs_effects_sequentially_and_passes_rng_and_dt():
    calls = []
    rng = object()
    mtq = Magnetorquer(
        (0, 0, 0), (1, 0, 0), 0.5, effects=[AddEffect(2, calls), AddEffect(-0.5, calls)]
    )
    assert mtq.apply_errors(1.0, 0.125, rng) == 2.5
    assert calls == [(1.0, 0.125, rng), (3.0, 0.125, rng)]


def test_apply_errors_without_effects_returns_input_unchanged():
    value = [0.1, -0.2, 0.3]
    mtq = Magnetorquer((0, 0, 0), (0, 0, 1), 0.4)
    assert mtq.apply_errors(value, 1.0) is value


def test_fields_are_mutable_instance_is_slotted_and_missing_arguments_fail():
    mtq = Magnetorquer((0, 0, 0), (1, 0, 0), 0.2)
    mtq.position = (1, 0, 0)
    mtq.max_dipole = 0.35
    mtq.misalignment = (0.01, -0.02, 0.03)
    assert mtq.position == (1, 0, 0)
    assert mtq.max_dipole == 0.35
    assert mtq.misalignment == (0.01, -0.02, 0.03)
    with pytest.raises(AttributeError):
        mtq.extra = True
    with pytest.raises(TypeError):
        Magnetorquer((0, 0, 0), (1, 0, 0))


def test_constructor_preserves_supplied_mutable_values_without_coercion():
    position = [1, 2, 3]
    effects = []
    mtq = Magnetorquer(position, [0, 0, 1], 0.2, effects=effects)
    assert mtq.position is position
    assert mtq.effects is effects
    position[0] = 8
    assert mtq.position == [8, 2, 3]
