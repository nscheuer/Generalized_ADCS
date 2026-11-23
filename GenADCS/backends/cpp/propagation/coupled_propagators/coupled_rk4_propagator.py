"""Fixed-step RK4 integration of the prepared coupled dynamics."""

import numpy as np
from numba import njit

from ._coupled_dynamics import SECONDS_PER_DAY, _prepare
from ._state_layout import pack_state, unpack_state

# Keep the _Runtime re-export so existing Numba disk caches can be unpickled.


from ._coupled import rk4_step as _rk4_step, project as _project_wheel_momentum

def _integrate_chunk(y, initial_jd, dt, full_steps, remainder, p, output, start, count):
    """Advance consecutive RK4 steps without returning to Python between them."""
    for row in range(start, start + count):
        elapsed_s = row * dt
        step_dt = remainder if row == full_steps else dt
        y = _rk4_step(y, initial_jd + elapsed_s / SECONDS_PER_DAY, step_dt, p)
        output[row + 1, :] = y
    return y

class CoupledRK4Propagator:
    """Fixed-step RK4 for Cartesian orbit and body-from-inertial quaternion.

    The state has quaternion, body rates, inertial position, velocity, and
    optional wheel momenta. Every contribution is reevaluated at each RK stage.
    """

    def __init__(self, dt):
        self.dt = float(dt)
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be finite and positive seconds")
        self._runtime = None
        self._configuration = None

    def reset_runtime(self):
        """Rebuild numerical parameters on the next step after configuration edits."""
        self._runtime = None
        self._configuration = None

    def integrate(self, state, satellite, environment, dynamics, tf, progress=False,
                  progress_chunk_steps=500, record_every=1):
        from ._coupled import integrate_rk4
        runtime = _prepare(satellite, environment, dynamics)
        y = pack_state(state, runtime.wheel_commands.size)
        return integrate_rk4(y, state.orbit.julian_date, self.dt, float(tf), runtime, progress, record_every)

    def step(self, state, satellite, environment, dynamics):
        if self._configuration != (id(satellite), id(environment), id(dynamics)):
            self._runtime = _prepare(satellite, environment, dynamics)
            self._configuration = (id(satellite), id(environment), id(dynamics))
        y = pack_state(state, self._runtime.wheel_commands.size)
        if self._runtime.wheel_commands.size:
            y[:] = _project_wheel_momentum(y, self._runtime)
        result = _rk4_step(y, state.orbit.julian_date, self.dt, self._runtime)
        return unpack_state(result, state, self.dt)
