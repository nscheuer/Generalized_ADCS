from types import SimpleNamespace

import numpy as np
import pytest

from GenADCS.backends.python.state.internal_state import (
    InternalState,
    WheelMomentumState,
)


def test_empty_default_is_independent() -> None:
    first, second = InternalState(), InternalState()
    assert first.wheel_momentum.shape == (0,)
    first.wheel_momentum = np.array([1.0])
    assert second.wheel_momentum.shape == (0,)


def test_momentum_is_finite_contiguous_float64_and_detached() -> None:
    source = np.array([1, 2], dtype=np.float32)
    state = InternalState(source)
    assert state.wheel_momentum.dtype == np.float64
    assert state.wheel_momentum.flags.c_contiguous
    source[0] = 99
    np.testing.assert_array_equal(state.wheel_momentum, [1, 2])


@pytest.mark.parametrize("momentum", [2, [[1, 2]], [1, np.inf], [np.nan]])
def test_rejects_invalid_momentum(momentum: object) -> None:
    with pytest.raises(ValueError, match="wheel_momentum must be a finite 1D array"):
        InternalState(momentum)


def test_wheel_state_copies_wheel_values_without_mutating_wheels() -> None:
    wheels = [SimpleNamespace(h=1.5), SimpleNamespace(h=-2.0)]
    subsystem = SimpleNamespace(reaction_wheels=wheels)
    state = WheelMomentumState(subsystem)
    np.testing.assert_array_equal(state.wheel_momentum, [1.5, -2])
    wheels[0].h = 9
    assert state.wheel_momentum[0] == 1.5
    assert isinstance(state, InternalState)


def test_wheel_state_explicit_values_and_conflicting_inputs() -> None:
    state = WheelMomentumState(wheel_momentum=[3, 4])
    np.testing.assert_array_equal(state.wheel_momentum, [3, 4])
    with pytest.raises(ValueError, match="provide subsystem or wheel_momentum"):
        WheelMomentumState(SimpleNamespace(reaction_wheels=[]), wheel_momentum=[])


def test_public_assignment_is_direct_and_can_replace_with_array() -> None:
    state = InternalState([1, 2])
    replacement = np.array([3.0, 4.0])
    state.wheel_momentum = replacement
    assert state.wheel_momentum is replacement
    state.wheel_momentum = [5, 6]
    assert state.wheel_momentum == [5, 6]


def test_wheel_state_none_sources_and_empty_subsystem() -> None:
    assert WheelMomentumState().wheel_momentum.shape == (0,)
    assert WheelMomentumState(
        SimpleNamespace(reaction_wheels=[])
    ).wheel_momentum.shape == (0,)


def test_wheel_state_propagates_bad_source_values_through_validation() -> None:
    with pytest.raises(ValueError, match="wheel_momentum"):
        WheelMomentumState(SimpleNamespace(reaction_wheels=[SimpleNamespace(h=np.inf)]))
