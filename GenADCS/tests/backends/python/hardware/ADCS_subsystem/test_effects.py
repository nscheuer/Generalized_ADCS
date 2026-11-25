"""Tests for ordered effect application and call forwarding."""

import pytest

from GenADCS.backends.python.hardware.ADCS_subsystem.effects import apply_effects


class RecordingEffect:
    def __init__(self, name, transform, calls):
        self.name = name
        self.transform = transform
        self.calls = calls

    def apply(self, value, dt, rng=None):
        self.calls.append((self.name, value, dt, rng))
        return self.transform(value)


def test_empty_effect_sequence_returns_original_object_unchanged():
    value = [1.0, 2.0]
    result = apply_effects(value, [], 0.5)
    assert result is value
    assert value == [1.0, 2.0]


def test_effects_apply_in_order_and_forward_each_intermediate_value():
    calls = []
    rng = object()
    effects = [
        RecordingEffect("double", lambda value: value * 2, calls),
        RecordingEffect("offset", lambda value: value + 3, calls),
        RecordingEffect("square", lambda value: value**2, calls),
    ]
    assert apply_effects(4.0, effects, 0.25, rng) == 121.0
    assert [call[0] for call in calls] == ["double", "offset", "square"]
    assert [call[1] for call in calls] == [4.0, 8.0, 11.0]
    assert all(call[2] == 0.25 and call[3] is rng for call in calls)


def test_effects_can_transform_arrays_without_mutating_original_when_returning_new_arrays():
    import numpy as np

    class AddVector:
        def apply(self, value, dt, rng=None):
            return np.asarray(value) + np.array([dt, -dt])

    original = np.array([1.0, 2.0])
    result = apply_effects(original, [AddVector()], 0.5)
    np.testing.assert_array_equal(original, [1.0, 2.0])
    np.testing.assert_array_equal(result, [1.5, 1.5])
    assert result is not original


def test_effects_accept_any_iterable_and_propagate_exceptions_without_running_later_effects():
    calls = []
    effects = (
        effect
        for effect in [
            RecordingEffect("before", lambda value: value + 1, calls),
            RecordingEffect(
                "failure",
                lambda value: (_ for _ in ()).throw(RuntimeError("effect failed")),
                calls,
            ),
            RecordingEffect("after", lambda value: value + 1, calls),
        ]
    )
    with pytest.raises(RuntimeError, match="effect failed"):
        apply_effects(0, effects, 1.0)
    assert [call[0] for call in calls] == ["before", "failure"]
