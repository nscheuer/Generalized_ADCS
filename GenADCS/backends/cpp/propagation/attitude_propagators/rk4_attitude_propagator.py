"""Fixed-step RK4 integration of attitude with a frozen orbit."""

import math

import numpy as np

from GenADCS.backends.cpp.propagation.coupled_propagators._coupled_dynamics import (
    _prepare,
)
from GenADCS.backends.cpp.propagation.coupled_propagators._state_layout import (
    pack_state,
)
from GenADCS.backends.cpp.state import AttitudeState, State, WheelMomentumState
from GenADCS.backends.cpp.propagation._separate import (
    integrate_attitude_rk4,
    rk4_attitude_step,
)


class RK4AttitudePropagator:
    def __init__(self, dt):
        self.dt = float(dt)
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be finite and positive seconds")

    def integrate(self, state, satellite, environment, dynamics, tf,
                  record_every=1):
        tf = float(tf)
        if not math.isfinite(tf) or tf < 0:
            raise ValueError("tf must be finite and nonnegative seconds")
        if isinstance(record_every, bool) or not isinstance(record_every, int) or record_every < 1:
            raise ValueError("record_every must be a positive integer")
        runtime = _prepare(satellite, environment, dynamics)
        y = pack_state(state, runtime.wheel_commands.size)
        times, values = integrate_attitude_rk4(
            y, state.orbit.julian_date, self.dt, tf, runtime, record_every
        )
        return times, values

    def step(self, state, satellite, environment, dynamics):
        runtime = _prepare(satellite, environment, dynamics)
        y = pack_state(state, runtime.wheel_commands.size)
        value = rk4_attitude_step(y, state.orbit.julian_date, self.dt, runtime)
        return State(
            AttitudeState(value[:4].copy(), value[4:7].copy()),
            state.orbit,
            WheelMomentumState(wheel_momentum=value[13:].copy()),
        )
