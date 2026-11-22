"""State packing for the C coupled backend."""
import numpy as np
from GenADCS.backends.cpp.state import AttitudeState, OrbitalState, State, WheelMomentumState

def pack_state(state, wheel_count):
    momentum = np.asarray(state.internal.wheel_momentum, dtype=np.float64)
    if momentum.size != wheel_count:
        raise ValueError(f"expected {wheel_count} wheel momenta, got {momentum.size}")
    return np.concatenate((state.attitude.quaternion_array, state.attitude.rates,
                           state.orbit.position, state.orbit.velocity, momentum))

def unpack_state(values, initial_state, elapsed_s):
    return State(
        AttitudeState(values[:4].copy(), values[4:7].copy()),
        OrbitalState.from_cartesian(values[7:10].copy(), values[10:13].copy(),
            initial_state.orbit.julian_date + elapsed_s / 86400.0,
            initial_state.orbit.ephemeris),
        WheelMomentumState(wheel_momentum=values[13:].copy()),
    )
