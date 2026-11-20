"""Adaptive coupled Dormand-Prince 8(5,3) propagation in Numba."""

import numpy as np
from numba import njit

from ._coupled import dop853_attempt as _dop853_attempt, derivative as _derivative, project as _project_wheel_momentum
from ._coupled_dynamics import SECONDS_PER_DAY, _prepare
from ._state_layout import pack_state, unpack_state


def _integrate_adaptive_chunk(y, f, t, h, initial_jd, tf, p, rtol, atol,
                              max_step, min_step, max_accepted_steps):
    """Store accepted states while all adaptive trial steps stay inside Numba."""
    capacity = min(max_accepted_steps + 1, 1024)
    capacity = max(capacity, 1)
    times = np.empty(capacity, dtype=np.float64)
    states = np.empty((capacity, y.size), dtype=np.float64)
    times[0] = t
    states[0, :] = y
    count = 1
    accepted = 0

    while t < tf and accepted < max_accepted_steps:
        remaining = tf - t
        step = min(h, max_step, remaining)
        if step <= 0.0 or t + step == t:
            raise ValueError("DOP853 step is too small to advance simulation time")
        if step < min_step and remaining > min_step:
            raise ValueError("DOP853 reached min_step before satisfying tolerances")

        candidate, next_f, error = _dop853_attempt(
            y, f, initial_jd + t / SECONDS_PER_DAY, step, p, rtol, atol,
        )
        if not np.isfinite(error):
            raise ValueError("DOP853 produced a nonfinite error estimate")
        if error <= 1.0:
            t += step
            y = candidate
            f = next_f
            if count == capacity:
                new_capacity = capacity * 2
                new_times = np.empty(new_capacity, dtype=np.float64)
                new_states = np.empty((new_capacity, y.size), dtype=np.float64)
                new_times[:count] = times[:count]
                new_states[:count, :] = states[:count, :]
                times = new_times
                states = new_states
                capacity = new_capacity
            times[count] = t
            states[count, :] = y
            count += 1
            accepted += 1
            factor = 10.0 if error == 0.0 else min(10.0, max(0.2, 0.9 * error ** (-0.125)))
            h = min(max_step, step * factor)
        else:
            h = step * max(0.2, 0.9 * error ** (-0.125))
            if h < min_step:
                raise ValueError("DOP853 reached min_step before satisfying tolerances")
    return times[:count], states[:count, :], y, f, t, h


class CoupledDOP853Propagator:
    """Adaptive DOP853 for the same coupled dynamics as RK4.

    ``dt`` is the initial proposed step in seconds. ``max_step`` caps later
    accepted steps. Error is controlled with componentwise absolute tolerance
    and a common relative tolerance.
    """

    def __init__(self, dt, rtol=1e-9, atol=1e-12, max_step=np.inf, min_step=1e-6):
        self.dt = float(dt)
        self.rtol = float(rtol)
        self.max_step = float(max_step)
        self.min_step = float(min_step)
        absolute = np.asarray(atol, dtype=np.float64)
        self._scalar_atol = absolute.ndim == 0
        if self._scalar_atol:
            absolute = np.full(13, float(absolute), dtype=np.float64)
        if absolute.ndim != 1 or absolute.size < 13:
            raise ValueError("atol must be a scalar or vector with at least 13 elements")
        self.atol = np.ascontiguousarray(absolute)
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive seconds")
        if not np.isfinite(self.rtol) or self.rtol <= 0.0:
            raise ValueError("rtol must be finite and positive")
        if not np.all(np.isfinite(self.atol)) or np.any(self.atol <= 0.0):
            raise ValueError("atol must contain finite positive values")
        if self.max_step <= 0.0 or np.isnan(self.max_step):
            raise ValueError("max_step must be positive")
        if not np.isfinite(self.min_step) or self.min_step <= 0.0 or self.min_step > self.max_step:
            raise ValueError("min_step must be finite, positive, and no greater than max_step")

    def integrate(self, state, satellite, environment, dynamics, tf, progress=False,
                  progress_chunk_steps=500):
        from ._coupled import integrate_dop853
        runtime = _prepare(satellite, environment, dynamics)
        y = pack_state(state, runtime.wheel_commands.size)
        atol = np.array([self.atol[0]]) if self._scalar_atol else self.atol
        return integrate_dop853(y, state.orbit.julian_date, self.dt, float(tf),
                                runtime, self.rtol, atol, self.max_step, self.min_step, progress)

    def step(self, state, satellite, environment, dynamics):
        """Advance by dt seconds using as many adaptive substeps as needed."""
        _, vectors = self.integrate(state, satellite, environment, dynamics, self.dt)
        result = vectors[-1]
        return unpack_state(result, state, self.dt)
