"""The numeric state layout shared by both compiled integrators."""

import numpy as np

from GenADCS.backends.python.state import State
from GenADCS.backends.python.state import (
    AttitudeState,
    OrbitalState,
    State,
    WheelMomentumState,
)

QUATERNION = slice(0, 4)
RATES = slice(4, 7)
POSITION = slice(7, 10)
VELOCITY = slice(10, 13)
WHEEL_MOMENTUM = slice(13, None)
BASE_SIZE = 13


def pack_state(state: State, wheel_count: int) -> np.ndarray:
    momentum = state.internal.wheel_momentum
    if momentum.size != wheel_count:
        raise ValueError(f"expected {wheel_count} wheel momenta, got {momentum.size}")
    return np.concatenate(
        (
            state.attitude.quaternion_array,
            state.attitude.rates,
            state.orbit.position,
            state.orbit.velocity,
            momentum,
        )
    )


def unpack_state(values: np.ndarray, initial_state: State, elapsed_s: float) -> State:
    return State(
        AttitudeState(values[QUATERNION].copy(), values[RATES].copy()),
        OrbitalState.from_cartesian(
            values[POSITION].copy(),
            values[VELOCITY].copy(),
            initial_state.orbit.julian_date + elapsed_s / 86400.0,
            initial_state.orbit.ephemeris,
        ),
        WheelMomentumState(wheel_momentum=values[WHEEL_MOMENTUM].copy()),
    )
