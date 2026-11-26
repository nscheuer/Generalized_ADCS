"""Single Numba loop for separate Keplerian orbit and RK4 attitude propagation."""

import math
from threading import Event, Thread

import numpy as np
from numba import njit

from GenADCS.backends.python.config import DynamicsConfig
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
)
from GenADCS.backends.python.propagation.attitude_propagators.rk4_attitude_propagator import (
    _rk4_attitude_step,
)
from GenADCS.backends.python.propagation.coupled_propagators._coupled_dynamics import (
    SECONDS_PER_DAY,
    RuntimeParameters,
    prepare_runtime,
    project_wheel_momentum,
)
from GenADCS.backends.python.propagation.coupled_propagators._state_layout import (
    pack_state,
)
from GenADCS.backends.python.propagation.orbital_propagators.keplerian_orbital_propagator import (
    keplerian_step,
)
from GenADCS.backends.python.state import State
from GenADCS.backends.python.propagation.attitude_propagators.reference_attitude_propagator import set_reference_attitude


@njit(cache=True, nogil=True)
def _integrate_separate(
    y: np.ndarray,
    initial_jd: float,
    dt: float,
    full_steps: int,
    remainder: float,
    runtime: RuntimeParameters,
    output: np.ndarray,
    record_every: int,
    sensor_every: int,
    sensor_config: SensorConfig,
    clean: np.ndarray,
    reading: np.ndarray,
    seed: int,
    estimator_kind: int,
    estimate_q: np.ndarray,
    estimate_w: np.ndarray,
    estimate_p: np.ndarray,
    acceleration_psd: np.ndarray,
    estimates: np.ndarray,
    estimator_flags: int,
    completed_steps: np.ndarray,
    progress_every: int,
    reference_mode: int,
) -> None:
    total_steps = full_steps + int(remainder > 0.0)
    for row in range(total_steps):
        elapsed = row * dt
        step_dt = remainder if row == full_steps else dt
        epoch_jd = initial_jd + elapsed / SECONDS_PER_DAY
        # Attitude sees the orbit at the start of this step. The orbit then
        # advances using its independent Keplerian two-body model.
        if reference_mode < 0:
            y = _rk4_attitude_step(y, epoch_jd, step_dt, runtime)
        position, velocity = keplerian_step(y[7:10], y[10:13], runtime.mu, step_dt)
        y[7:10] = position
        y[10:13] = velocity
        if reference_mode >= 0:
            set_reference_attitude(y, epoch_jd + step_dt / SECONDS_PER_DAY, reference_mode, runtime)
        completed = row + 1
        if (
            sensor_every > 0
            and completed <= full_steps
            and completed % sensor_every == 0
        ):
            index = completed // sensor_every
            sample_sensors(
                y,
                initial_jd + completed * dt / SECONDS_PER_DAY,
                runtime,
                sensor_config,
                clean[index],
                reading[index],
                seed,
                index,
            )
            if estimator_kind:
                update_estimator(
                    estimator_kind,
                    estimate_q,
                    estimate_w,
                    estimate_p,
                    acceleration_psd,
                    dt * sensor_every,
                    reading[index],
                    sensor_config,
                    estimator_flags,
                    y[:4],
                    y[4:7],
                    estimates[index, 43:],
                )
                if estimator_flags & 1:
                    estimates[index, :4] = estimate_q
                if estimator_flags & 2:
                    estimates[index, 4:7] = estimate_w
                if estimator_flags & 1024:
                    estimates[index, 7:43] = estimate_p.ravel()
        if completed % record_every == 0:
            output[completed // record_every] = y
        elif completed == total_steps:
            output[-1] = y
        if progress_every > 0 and (
            completed % progress_every == 0 or completed == total_steps
        ):
            completed_steps[0] = completed


def integrate_separate(
    state: State,
    satellite: Satellite,
    environment: Environment,
    dynamics: DynamicsConfig,
    dt: float,
    tf: float,
    record_every: int,
    sensor_every: int,
    seed: int,
    estimator_flags: int,
    progress: bool = False,
    progress_chunk_steps: int = 500,
    reference_mode: int = -1,
) -> tuple[np.ndarray, ...]:
    """Prepare once, execute every step in one JIT call, return numeric samples."""
    tf = float(tf)
    if not math.isfinite(tf) or tf < 0:
        raise ValueError("tf must be finite and nonnegative seconds")
    if isinstance(record_every, bool) or record_every < 1:
        raise ValueError("record_every must be a positive integer")
    if (
        isinstance(progress_chunk_steps, bool)
        or not isinstance(progress_chunk_steps, int)
        or progress_chunk_steps < 1
    ):
        raise ValueError("progress_chunk_steps must be a positive integer")
    runtime = prepare_runtime(satellite, environment, dynamics)
    y = pack_state(state, runtime.wheel_commands.size)
    if runtime.wheel_commands.size:
        project_wheel_momentum(y, runtime)
    full_steps = math.floor(tf / dt)
    remainder = tf - full_steps * dt
    if remainder <= 1e-12 * max(1.0, tf):
        remainder = 0.0
    total_steps = full_steps + int(remainder > 0.0)
    sampled_steps = np.arange(0, total_steps + 1, record_every, dtype=np.int64)
    elapsed = np.minimum(sampled_steps.astype(np.float64) * dt, tf)
    if sampled_steps[-1] != total_steps:
        elapsed = np.append(elapsed, tf)
    else:
        elapsed[-1] = tf
    output = np.empty((elapsed.size, y.size), dtype=np.float64)
    if reference_mode >= 0:
        set_reference_attitude(y, state.orbit.julian_date, reference_mode, runtime)
    output[0] = y
    sensor_config = prepare_sensors(
        satellite.adcs.sensors if sensor_every else [], environment
    )
    width = max(1, int(np.sum(sensor_config[-2])))
    sensor_rows = full_steps // sensor_every + 1 if sensor_every else 1
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
            runtime,
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
            )
            if estimator_flags & 1:
                estimates[0, :4] = estimate_q
            if estimator_flags & 2:
                estimates[0, 4:7] = estimate_w
            if estimator_flags & 1024:
                estimates[0, 7:43] = estimate_p.ravel()
    if total_steps:
        completed_steps = np.zeros(1, dtype=np.int64)
        args = (
            y,
            state.orbit.julian_date,
            dt,
            full_steps,
            remainder,
            runtime,
            output,
            record_every,
            sensor_every,
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
            completed_steps,
        )
        if progress:
            from tqdm import tqdm

            # Compile before starting the monitor so the step bar measures
            # propagation, just as in the coupled RK4 implementation.
            with tqdm(total=1, desc="Compiling Numba kernels", unit="stage") as bar:
                _integrate_separate(*args[:3], 0, 0.0, *args[5:], progress_chunk_steps, reference_mode)
                bar.update(1)

            finished = Event()

            def monitor() -> None:
                with tqdm(
                    total=total_steps, desc="Running Keplerian + RK4", unit="step"
                ) as bar:
                    while not finished.wait(0.1):
                        current = int(completed_steps[0])
                        if current > bar.n:
                            bar.update(current - bar.n)
                    current = int(completed_steps[0])
                    if current > bar.n:
                        bar.update(current - bar.n)

            thread = Thread(target=monitor, daemon=True)
            thread.start()
            try:
                _integrate_separate(*args, progress_chunk_steps, reference_mode)
            finally:
                finished.set()
                thread.join()
        else:
            _integrate_separate(*args, 0, reference_mode)
    return (
        (elapsed, output, clean, reading, estimates)
        if estimator_kind
        else (elapsed, output, clean, reading) if sensor_every else (elapsed, output)
    )
