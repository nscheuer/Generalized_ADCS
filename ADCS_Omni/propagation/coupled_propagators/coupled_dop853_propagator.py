"""Adaptive coupled Dormand-Prince 8(5,3) propagation in Numba."""

import numpy as np
from numba import njit

from ADCS_Omni.state import AttitudeState, InternalState, OrbitalState, State

from ._dop853_coefficients import A, B, C, E3, E5
from .coupled_rk4_propagator import SECONDS_PER_DAY, _derivative, _prepare


@njit(cache=True)
def _dop853_attempt(y, f, epoch_jd, h, p, rtol, atol):
    """One DOP853 trial, including the embedded 5/3 error estimate."""
    stages = np.empty((13, 13), dtype=np.float64)
    stages[0, :] = f
    for stage in range(1, 12):
        trial = y.copy()
        for previous in range(stage):
            coefficient = h * A[stage, previous]
            if coefficient != 0.0:
                for component in range(13):
                    trial[component] += coefficient * stages[previous, component]
        stages[stage, :] = _derivative(trial, epoch_jd + C[stage] * h / SECONDS_PER_DAY, p)

    candidate = y.copy()
    for stage in range(12):
        coefficient = h * B[stage]
        if coefficient != 0.0:
            for component in range(13):
                candidate[component] += coefficient * stages[stage, component]
    candidate[0:4] /= np.sqrt(np.dot(candidate[0:4], candidate[0:4]))
    stages[12, :] = _derivative(candidate, epoch_jd + h / SECONDS_PER_DAY, p)

    norm5_squared = 0.0
    norm3_squared = 0.0
    for component in range(13):
        scale = atol[component] + rtol * max(abs(y[component]), abs(candidate[component]))
        err5 = 0.0
        err3 = 0.0
        for stage in range(13):
            err5 += E5[stage] * stages[stage, component]
            err3 += E3[stage] * stages[stage, component]
        norm5_squared += (err5 / scale) ** 2
        norm3_squared += (err3 / scale) ** 2
    if norm5_squared == 0.0 and norm3_squared == 0.0:
        error_norm = 0.0
    else:
        error_norm = abs(h) * norm5_squared / np.sqrt((norm5_squared + 0.01 * norm3_squared) * 13.0)
    return candidate, stages[12, :].copy(), error_norm


@njit(cache=True)
def _integrate_adaptive_chunk(y, f, t, h, initial_jd, tf, p, rtol, atol,
                              max_step, min_step, max_accepted_steps):
    """Store accepted states while all adaptive trial steps stay inside Numba."""
    capacity = min(max_accepted_steps + 1, 1024)
    capacity = max(capacity, 1)
    times = np.empty(capacity, dtype=np.float64)
    states = np.empty((capacity, 13), dtype=np.float64)
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
                new_states = np.empty((new_capacity, 13), dtype=np.float64)
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
    """Adaptive DOP853 for the same passive coupled dynamics as RK4.

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
        if absolute.ndim == 0:
            absolute = np.full(13, float(absolute), dtype=np.float64)
        if absolute.shape != (13,):
            raise ValueError("atol must be a scalar or 13-element vector")
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
        tf = float(tf)
        if not np.isfinite(tf) or tf < 0.0:
            raise ValueError("tf must be finite and nonnegative seconds")
        if progress and (isinstance(progress_chunk_steps, bool)
                         or not isinstance(progress_chunk_steps, int)
                         or progress_chunk_steps <= 0):
            raise ValueError("progress_chunk_steps must be a positive integer")
        if state.internal.wheel_momentum.size:
            raise NotImplementedError("coupled passive propagation does not support wheel momentum")

        p = _prepare(satellite, environment, dynamics)
        y = np.concatenate((state.attitude.quaternion_array, state.attitude.rates,
                            state.orbit.position, state.orbit.velocity))
        if tf == 0.0:
            return np.array([0.0]), y.reshape(1, 13)

        t = 0.0
        h = min(self.dt, self.max_step)
        f = _derivative(y, state.orbit.julian_date, p)
        if progress:
            from tqdm import tqdm

            with tqdm(total=1, desc="Compiling Numba kernels", unit="stage") as bar:
                _integrate_adaptive_chunk(y, f, t, h, state.orbit.julian_date, tf,
                                          p, self.rtol, self.atol, self.max_step,
                                          self.min_step, 0)
                bar.update(1)
            time_parts = [np.array([0.0])]
            state_parts = [y.reshape(1, 13)]
            with tqdm(total=tf, desc="Running coupled DOP853", unit="s") as bar:
                while t < tf:
                    times, states, y, f, new_t, h = _integrate_adaptive_chunk(
                        y, f, t, h, state.orbit.julian_date, tf, p,
                        self.rtol, self.atol, self.max_step, self.min_step,
                        progress_chunk_steps,
                    )
                    time_parts.append(times[1:])
                    state_parts.append(states[1:, :])
                    bar.update(new_t - t)
                    t = new_t
            return np.concatenate(time_parts), np.concatenate(state_parts)

        times, states, _, _, _, _ = _integrate_adaptive_chunk(
            y, f, t, h, state.orbit.julian_date, tf, p,
            self.rtol, self.atol, self.max_step, self.min_step,
            2_147_483_647,
        )
        return times, states

    def step(self, state, satellite, environment, dynamics):
        """Advance by dt seconds using as many adaptive substeps as needed."""
        _, vectors = self.integrate(state, satellite, environment, dynamics, self.dt)
        result = vectors[-1]
        return State(
            AttitudeState(result[0:4].copy(), result[4:7].copy()),
            OrbitalState.from_cartesian(result[7:10].copy(), result[10:13].copy(),
                                        state.orbit.julian_date + self.dt / SECONDS_PER_DAY,
                                        state.orbit.ephemeris),
            InternalState(),
        )
