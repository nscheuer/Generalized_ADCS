from types import SimpleNamespace

import numpy as np
import pytest


def test_empty_and_finite_momentum(state_backend):
    assert np.asarray(state_backend.InternalState().wheel_momentum).size == 0
    source = np.array([1.5, -2], dtype=np.float32)
    state = state_backend.InternalState(source)
    np.testing.assert_array_equal(state.wheel_momentum, [1.5, -2])
    source[0] = 9
    assert np.asarray(state.wheel_momentum)[0] == 1.5


@pytest.mark.parametrize("momentum", (1.0, [[1, 2]], [np.nan], [np.inf]))
def test_invalid_momentum_rejected(state_backend, momentum):
    with pytest.raises(ValueError):
        state_backend.InternalState(momentum)


def test_wheel_momentum_from_subsystem(state_backend):
    wheels = [SimpleNamespace(h=1.25), SimpleNamespace(h=-0.5)]
    subsystem = SimpleNamespace(reaction_wheels=wheels)
    state = state_backend.WheelMomentumState(subsystem)
    np.testing.assert_array_equal(state.wheel_momentum, [1.25, -0.5])
    wheels[0].h = 99
    assert np.asarray(state.wheel_momentum)[0] == 1.25


def test_wheel_sources_are_mutually_exclusive(state_backend):
    with pytest.raises(ValueError):
        state_backend.WheelMomentumState(
            SimpleNamespace(reaction_wheels=[]), wheel_momentum=[]
        )
