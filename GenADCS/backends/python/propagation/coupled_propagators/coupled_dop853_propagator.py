"""Adaptive coupled Dormand-Prince 8(5,3) propagation in Numba."""

import numpy as np
from numba import njit
from numpy.typing import ArrayLike

from GenADCS.backends.python.config.dynamics_config import DynamicsConfig
from GenADCS.backends.python.environment import Environment
from GenADCS.backends.python.hardware import Satellite
from GenADCS.backends.python.hardware.ADCS_subsystem.estimator.runtime import (
    prepare_estimator,
    update_estimator,
)
from GenADCS.backends.python.hardware.ADCS_subsystem.sensors.sensor_sampling import (
    SensorConfig,
    prepare_sensors,
    sample_sensors,
    attitude_references,
)
from GenADCS.backends.python.state import State

from ._coupled_dynamics import (
    SECONDS_PER_DAY,
)
from ._coupled_dynamics import (
    RuntimeParameters as _Runtime,
)
from ._coupled_dynamics import (
    derivative as _derivative,
)
from ._coupled_dynamics import (
    prepare_runtime as _prepare,
)
from ._coupled_dynamics import (
    project_wheel_momentum as _project_wheel_momentum,
)
from ._dop853_coefficients import E3, E5, A, B, C
from ._state_layout import pack_state, unpack_state

DEFAULT_SEED = 0


@njit(cache=True)
def _dop853_attempt(
    y: np.ndarray,
    f: np.ndarray,
    epoch_jd: float,
    h: float,
    p: _Runtime,
    rtol: float,
    atol: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """One DOP853 trial, including the embedded 5/3 error estimate."""
    components = y.size
    stages = np.empty((13, components), dtype=np.float64)
    stages[0, :] = f
    for stage in range(1, 12):
        trial = y.copy()
        for previous in range(stage):
            coefficient = h * A[stage, previous]
            if coefficient != 0.0:
                for component in range(components):
                    trial[component] += coefficient * stages[previous, component]
        stages[stage, :] = _derivative(
            trial, epoch_jd + C[stage] * h / SECONDS_PER_DAY, p
        )

    candidate = y.copy()
    for stage in range(12):
        coefficient = h * B[stage]
        if coefficient != 0.0:
            for component in range(components):
                candidate[component] += coefficient * stages[stage, component]
    candidate[0:4] /= np.sqrt(np.dot(candidate[0:4], candidate[0:4]))
    if p.wheel_commands.size:
        _project_wheel_momentum(candidate, p)
    stages[12, :] = _derivative(candidate, epoch_jd + h / SECONDS_PER_DAY, p)

    norm5_squared = 0.0
    norm3_squared = 0.0
    for component in range(components):
        scale = atol[component] + rtol * max(
            abs(y[component]), abs(candidate[component])
        )
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
        error_norm = (
            abs(h)
            * norm5_squared
            / np.sqrt((norm5_squared + 0.01 * norm3_squared) * components)
        )
    return candidate, stages[12, :].copy(), error_norm


@njit(cache=True)
def _integrate_adaptive_chunk(
    y: np.ndarray,
    f: np.ndarray,
    t: float,
    h: float,
    initial_jd: float,
    tf: float,
    p: _Runtime,
    rtol: float,
    atol: np.ndarray,
    max_step: float,
    min_step: float,
    max_accepted_steps: int,
    sample_interval: float = 0.0,
    sensor_interval: float = 0.0,
    sensor_config: SensorConfig | None = None,
    clean: np.ndarray | None = None,
    reading: np.ndarray | None = None,
    seed: int = DEFAULT_SEED,
    estimator_kind: int = 0,
    estimate_q: np.ndarray | None = None,
    estimate_w: np.ndarray | None = None,
    estimate_p: np.ndarray | None = None,
    acceleration_psd: np.ndarray | None = None,
    estimates: np.ndarray | None = None,
    estimator_flags: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Store accepted states while all adaptive trial steps stay inside Numba."""
    capacity = min(max_accepted_steps + 1, 1024)
    capacity = max(capacity, 1)
    times = np.empty(capacity, dtype=np.float64)
    states = np.empty((capacity, y.size), dtype=np.float64)
    times[0] = t
    states[0, :] = y
    count = 1
    accepted = 0
    next_sample = (
        (np.floor(t / sample_interval + 1e-10) + 1.0) * sample_interval
        if sample_interval > 0.0
        else 0.0
    )

    while t < tf and accepted < max_accepted_steps:
        remaining = tf - t
        step = min(h, max_step, remaining)
        if sample_interval > 0.0 and next_sample < tf:
            step = min(step, next_sample - t)
        if step <= 0.0 or t + step == t:
            raise ValueError("DOP853 step is too small to advance simulation time")
        boundary_limited = (
            sample_interval > 0.0 and next_sample < tf and next_sample - t <= min_step
        )
        if step < min_step and remaining > min_step and not boundary_limited:
            raise ValueError("DOP853 reached min_step before satisfying tolerances")

        candidate, next_f, error = _dop853_attempt(
            y,
            f,
            initial_jd + t / SECONDS_PER_DAY,
            step,
            p,
            rtol,
            atol,
        )
        if not np.isfinite(error):
            raise ValueError("DOP853 produced a nonfinite error estimate")
        if error <= 1.0:
            t += step
            y = candidate
            f = next_f
            save = sample_interval == 0.0 or abs(t - next_sample) <= 1e-10 * max(
                1.0, next_sample
            )
            if save and count == capacity:
                new_capacity = capacity * 2
                new_times = np.empty(new_capacity, dtype=np.float64)
                new_states = np.empty((new_capacity, y.size), dtype=np.float64)
                new_times[:count] = times[:count]
                new_states[:count, :] = states[:count, :]
                times = new_times
                states = new_states
                capacity = new_capacity
            if save:
                if sensor_interval > 0.0:
                    assert clean is not None and reading is not None
                    assert sensor_config is not None
                    sensor_index = int(np.round(t / sensor_interval))
                    if sensor_index < clean.shape[0] and abs(
                        t - sensor_index * sensor_interval
                    ) < 1e-9 * max(1.0, t):
                        sample_sensors(
                            y,
                            initial_jd + t / SECONDS_PER_DAY,
                            p,
                            sensor_config,
                            clean[sensor_index],
                            reading[sensor_index],
                            seed,
                            sensor_index,
                        )
                        if estimator_kind:
                            assert estimate_q is not None and estimate_w is not None
                            assert estimate_p is not None and acceleration_psd is not None
                            assert estimates is not None
                            update_estimator(
                                estimator_kind,
                                estimate_q,
                                estimate_w,
                                estimate_p,
                                acceleration_psd,
                                sensor_interval,
                                reading[sensor_index],
                                sensor_config,
                                estimator_flags,
                                y[:4],
                                y[4:7],
                                estimates[sensor_index, 43:],
                                references=attitude_references(y[7:10], initial_jd + t / SECONDS_PER_DAY, p, sensor_config),
                            )
                            if estimator_flags & 1:
                                estimates[sensor_index, :4] = estimate_q
                            if estimator_flags & 2:
                                estimates[sensor_index, 4:7] = estimate_w
                            if estimator_flags & 1024:
                                if estimator_kind == 2:
                                    estimates[sensor_index, 7:56] = estimate_p.ravel()
                                else:
                                    estimates[sensor_index, 7:43] = estimate_p.ravel()
                times[count] = t
                states[count, :] = y
                count += 1
                if sample_interval > 0.0:
                    next_sample += sample_interval
            accepted += 1
            factor = (
                10.0 if error == 0.0 else min(10.0, max(0.2, 0.9 * error ** (-0.125)))
            )
            h = min(max_step, step * factor)
        else:
            h = step * max(0.2, 0.9 * error ** (-0.125))
            if h < min_step and not boundary_limited:
                raise ValueError("DOP853 reached min_step before satisfying tolerances")
    return times[:count], states[:count, :], y, f, t, h


class CoupledDOP853Propagator:
    """Adaptive DOP853 for the same coupled dynamics as RK4.

    ``dt`` is the initial proposed step in seconds. ``max_step`` caps later
    accepted steps. Error is controlled with componentwise absolute tolerance
    and a common relative tolerance.
    """

    def __init__(
        self,
        dt: float,
        rtol: float = 1e-9,
        atol: ArrayLike = 1e-12,
        max_step: float = np.inf,
        min_step: float = 1e-6,
    ) -> None:
        self.dt = float(dt)
        self.rtol = float(rtol)
        self.max_step = float(max_step)
        self.min_step = float(min_step)
        absolute = np.asarray(atol, dtype=np.float64)
        self._scalar_atol = absolute.ndim == 0
        if self._scalar_atol:
            absolute = np.full(13, float(absolute), dtype=np.float64)
        if absolute.ndim != 1 or absolute.size < 13:
            raise ValueError(
                "atol must be a scalar or vector with at least 13 elements"
            )
        self.atol = np.ascontiguousarray(absolute)
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive seconds")
        if not np.isfinite(self.rtol) or self.rtol <= 0.0:
            raise ValueError("rtol must be finite and positive")
        if not np.all(np.isfinite(self.atol)) or np.any(self.atol <= 0.0):
            raise ValueError("atol must contain finite positive values")
        if self.max_step <= 0.0 or np.isnan(self.max_step):
            raise ValueError("max_step must be positive")
        if (
            not np.isfinite(self.min_step)
            or self.min_step <= 0.0
            or self.min_step > self.max_step
        ):
            raise ValueError(
                "min_step must be finite, positive, and no greater than max_step"
            )

    def integrate(
        self,
        state: State,
        satellite: Satellite,
        environment: Environment,
        dynamics: DynamicsConfig,
        tf: float,
        progress: bool = False,
        progress_chunk_steps: int = 500,
        record_every: int | None = None,
        sensor_every: int = 0,
        seed: int = 0,
        estimator_flags: int = 0,
    ) -> tuple[np.ndarray, ...]:
        tf = float(tf)
        if not np.isfinite(tf) or tf < 0.0:
            raise ValueError("tf must be finite and nonnegative seconds")
        if progress and (
            isinstance(progress_chunk_steps, bool)
            or progress_chunk_steps <= 0
        ):
            raise ValueError("progress_chunk_steps must be a positive integer")
        p = _prepare(satellite, environment, dynamics)
        y = pack_state(state, p.wheel_commands.size)
        if p.wheel_commands.size:
            _project_wheel_momentum(y, p)
        atol = np.full(y.size, self.atol[0]) if self._scalar_atol else self.atol
        if atol.shape != (y.size,):
            raise ValueError(f"atol must contain {y.size} values for this state")
        sensor_config = prepare_sensors(
            satellite.adcs.sensors if sensor_every else [], environment
        )
        width = max(1, int(np.sum(sensor_config[-2])))
        sensor_rows = (
            int(np.floor(tf / (self.dt * sensor_every) + 1e-10)) + 1
            if sensor_every
            else 1
        )
        clean = np.full((sensor_rows, width), np.nan)
        reading = np.full_like(clean, np.nan)
        estimator_kind, estimate_q, estimate_w, estimate_p, acceleration_psd = (
            prepare_estimator(satellite.adcs.estimator)
        )
        estimates = np.full((sensor_rows, 96), np.nan)
        if sensor_every:
            sample_sensors(
                y,
                state.orbit.julian_date,
                p,
                sensor_config,
                clean[0],
                reading[0],
                seed,
                0,
            )
            if estimator_kind:
                update_estimator(
                    estimator_kind,
                    estimate_q,
                    estimate_w,
                    estimate_p,
                    acceleration_psd,
                    0.0,
                    reading[0],
                    sensor_config,
                    estimator_flags,
                    y[:4],
                    y[4:7],
                    estimates[0, 43:],
                    references=attitude_references(y[7:10], state.orbit.julian_date, p, sensor_config),
                )
                if estimator_flags & 1:
                    estimates[0, :4] = estimate_q
                if estimator_flags & 2:
                    estimates[0, 4:7] = estimate_w
                if estimator_flags & 1024:
                    if estimator_kind == 2:
                        estimates[0, 7:56] = estimate_p.ravel()
                    else:
                        estimates[0, 7:43] = estimate_p.ravel()
        if tf == 0.0:
            result = (np.array([0.0]), y.reshape(1, y.size))
            return (
                (*result, clean, reading, estimates)
                if estimator_kind
                else (*result, clean, reading)
                if sensor_every
                else result
            )

        if record_every is not None and (
            isinstance(record_every, bool)
            or record_every < 1
        ):
            raise ValueError("record_every must be a positive integer")
        sample_interval = 0.0 if record_every is None else record_every * self.dt

        t = 0.0
        h = min(self.dt, self.max_step)
        f = _derivative(y, state.orbit.julian_date, p)
        if progress:
            from tqdm import tqdm

            with tqdm(total=1, desc="Compiling Numba kernels", unit="stage") as bar:
                _integrate_adaptive_chunk(
                    y,
                    f,
                    t,
                    h,
                    state.orbit.julian_date,
                    tf,
                    p,
                    self.rtol,
                    atol,
                    self.max_step,
                    self.min_step,
                    0,
                    sample_interval,
                    self.dt * sensor_every,
                    sensor_config,
                    clean,
                    reading,
                    seed,
                    estimator_kind,
                    estimate_q,
                    estimate_w,
                    estimate_p,
                    acceleration_psd,
                    estimates,
                    estimator_flags,
                )
                bar.update(1)
            time_parts = [np.array([0.0])]
            state_parts = [y.reshape(1, y.size)]
            main_step_count = int(np.ceil(tf / self.dt))
            completed_main_steps = 0
            with tqdm(
                total=main_step_count,
                desc="Running coupled DOP853",
                unit="step",
                dynamic_ncols=True,
                mininterval=0.1,
                miniters=1,
            ) as bar:
                while t < tf:
                    times, states, y, f, new_t, h = _integrate_adaptive_chunk(
                        y,
                        f,
                        t,
                        h,
                        state.orbit.julian_date,
                        tf,
                        p,
                        self.rtol,
                        atol,
                        self.max_step,
                        self.min_step,
                        progress_chunk_steps,
                        sample_interval,
                        self.dt * sensor_every,
                        sensor_config,
                        clean,
                        reading,
                        seed,
                        estimator_kind,
                        estimate_q,
                        estimate_w,
                        estimate_p,
                        acceleration_psd,
                        estimates,
                        estimator_flags,
                    )
                    time_parts.append(times[1:])
                    state_parts.append(states[1:, :])
                    t = new_t
                    current_main_steps = min(
                        main_step_count, int(np.floor(t / self.dt + 1e-12))
                    )
                    if t >= tf:
                        current_main_steps = main_step_count
                    bar.update(current_main_steps - completed_main_steps)
                    completed_main_steps = current_main_steps
            times = np.concatenate(time_parts)
            states = np.concatenate(state_parts)
            if sample_interval and abs(times[-1] - tf) > 1e-10 * max(1.0, tf):
                times = np.append(times, tf)
                states = np.vstack((states, y))
            return (
                (times, states, clean, reading, estimates)
                if estimator_kind
                else (times, states, clean, reading)
                if sensor_every
                else (times, states)
            )

        times, states, y, _, _, _ = _integrate_adaptive_chunk(
            y,
            f,
            t,
            h,
            state.orbit.julian_date,
            tf,
            p,
            self.rtol,
            atol,
            self.max_step,
            self.min_step,
            2_147_483_647,
            sample_interval,
            self.dt * sensor_every,
            sensor_config,
            clean,
            reading,
            seed,
            estimator_kind,
            estimate_q,
            estimate_w,
            estimate_p,
            acceleration_psd,
            estimates,
            estimator_flags,
        )
        if sample_interval and abs(times[-1] - tf) > 1e-10 * max(1.0, tf):
            times = np.append(times, tf)
            states = np.vstack((states, y))
        return (
            (times, states, clean, reading, estimates)
            if estimator_kind
            else (times, states, clean, reading)
            if sensor_every
            else (times, states)
        )

    def step(
        self,
        state: State,
        satellite: Satellite,
        environment: Environment,
        dynamics: DynamicsConfig,
    ) -> State:
        """Advance by dt seconds using as many adaptive substeps as needed."""
        _, vectors = self.integrate(state, satellite, environment, dynamics, self.dt)
        result = vectors[-1]
        return unpack_state(result, state, self.dt)
