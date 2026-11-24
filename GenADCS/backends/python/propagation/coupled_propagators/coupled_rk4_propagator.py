"""Fixed-step RK4 integration of the prepared coupled dynamics."""

import numpy as np
from numba import njit

from ._coupled_dynamics import SECONDS_PER_DAY, _Runtime, _derivative, _prepare, _project_wheel_momentum
from ._state_layout import pack_state, unpack_state
from GenADCS.backends.python.hardware.ADCS_subsystem.sensors.sensor_sampling import sample_sensors, prepare_sensors
from GenADCS.backends.python.hardware.ADCS_subsystem.estimator.runtime import prepare_estimator, update_estimator

# Keep the _Runtime re-export so existing Numba disk caches can be unpickled.


@njit(cache=True)
def _rk4_step(y, epoch_jd, dt, p):
    k1 = _derivative(y, epoch_jd, p)
    trial = y + 0.5 * dt * k1
    k2 = _derivative(trial, epoch_jd + 0.5 * dt / SECONDS_PER_DAY, p)
    trial = y + 0.5 * dt * k2
    k3 = _derivative(trial, epoch_jd + 0.5 * dt / SECONDS_PER_DAY, p)
    trial = y + dt * k3
    k4 = _derivative(trial, epoch_jd + dt / SECONDS_PER_DAY, p)
    result = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    result[0:4] /= np.sqrt(np.dot(result[0:4], result[0:4]))
    if p.wheel_commands.size:
        _project_wheel_momentum(result, p)
    return result


@njit(cache=True)
def _integrate_chunk(y, initial_jd, dt, full_steps, remainder, p, output, start, count, record_every,
                     sensor_every, sensor_config, clean, reading, seed,
                     estimator_kind, estimate_q, estimate_w, estimate_p, acceleration_psd, estimates, estimator_flags):
    """Advance consecutive RK4 steps without returning to Python between them."""
    for row in range(start, start + count):
        elapsed_s = row * dt
        step_dt = remainder if row == full_steps else dt
        y = _rk4_step(y, initial_jd + elapsed_s / SECONDS_PER_DAY, step_dt, p)
        completed = row + 1
        if sensor_every > 0 and completed <= full_steps and completed % sensor_every == 0:
            index = completed // sensor_every
            sample_sensors(y, initial_jd + completed * dt / SECONDS_PER_DAY, p,
                           sensor_config, clean[index], reading[index], seed, index)
            if estimator_kind:
                update_estimator(estimator_kind, estimate_q, estimate_w, estimate_p,
                                 acceleration_psd, dt * sensor_every, reading[index], sensor_config,
                                 estimator_flags, y[:4], y[4:7], estimates[index, 43:])
                if estimator_flags & 1:
                    estimates[index, :4] = estimate_q
                if estimator_flags & 2:
                    estimates[index, 4:7] = estimate_w
                if estimator_flags & 1024:
                    estimates[index, 7:43] = estimate_p.ravel()
        if completed % record_every == 0:
            output[completed // record_every, :] = y
        elif completed == full_steps + int(remainder > 0.0):
            output[-1, :] = y
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
                  progress_chunk_steps=500, record_every=1, sensor_every=0, seed=0, estimator_flags=0):
        """Return elapsed seconds and numeric states for the complete interval.

        Numeric state columns are quaternion (0:4), body rates (4:7),
        inertial position (7:10), inertial velocity (10:13), and wheel momenta
        (13:).
        """
        tf = float(tf)
        if not np.isfinite(tf) or tf < 0.0:
            raise ValueError("tf must be finite and nonnegative seconds")
        if progress and (isinstance(progress_chunk_steps, bool)
                         or not isinstance(progress_chunk_steps, int)
                         or progress_chunk_steps <= 0):
            raise ValueError("progress_chunk_steps must be a positive integer")
        runtime = _prepare(satellite, environment, dynamics)
        y = pack_state(state, runtime.wheel_commands.size)
        if runtime.wheel_commands.size:
            _project_wheel_momentum(y, runtime)
        full_steps = int(np.floor(tf / self.dt))
        remainder = tf - full_steps * self.dt
        if remainder <= 1e-12 * max(1.0, tf):
            remainder = 0.0
        step_count = full_steps + int(remainder > 0.0)
        if isinstance(record_every, bool) or not isinstance(record_every, int) or record_every < 1:
            raise ValueError("record_every must be a positive integer")
        sampled_steps = np.arange(0, step_count + 1, record_every, dtype=np.int64)
        terminal_extra = sampled_steps[-1] != step_count
        elapsed = np.minimum(sampled_steps.astype(np.float64) * self.dt, tf)
        if terminal_extra:
            elapsed = np.append(elapsed, tf)
        else:
            elapsed[-1] = tf
        output = np.empty((len(elapsed), y.size), dtype=np.float64)
        output[0, :] = y
        sensor_config = prepare_sensors(satellite.adcs.sensors if sensor_every else [], environment)
        width = max(1, int(np.sum(sensor_config[-2])))
        sensor_rows = full_steps // sensor_every + 1 if sensor_every else 1
        clean = np.full((sensor_rows, width), np.nan)
        reading = np.full_like(clean, np.nan)
        estimator_kind, estimate_q, estimate_w, estimate_p, acceleration_psd = prepare_estimator(satellite.adcs.estimator)
        estimates = np.full((sensor_rows, 96), np.nan)
        if sensor_every:
            sample_sensors(y, state.orbit.julian_date, runtime, sensor_config, clean[0], reading[0], seed, 0)
            if estimator_kind:
                update_estimator(estimator_kind, estimate_q, estimate_w, estimate_p,
                                 acceleration_psd, 0.0, reading[0], sensor_config,
                                 estimator_flags, y[:4], y[4:7], estimates[0, 43:])
                if estimator_flags & 1:
                    estimates[0, :4] = estimate_q
                if estimator_flags & 2:
                    estimates[0, 4:7] = estimate_w
                if estimator_flags & 1024:
                    estimates[0, 7:43] = estimate_p.ravel()

        if progress and step_count:
            from tqdm import tqdm

            with tqdm(total=1, desc="Compiling Numba kernels", unit="stage") as bar:
                _integrate_chunk(y, state.orbit.julian_date, self.dt, full_steps,
                                 remainder, runtime, output, 0, 0, record_every,
                                 sensor_every, sensor_config, clean, reading, seed,
                                 estimator_kind, estimate_q, estimate_w, estimate_p, acceleration_psd, estimates, estimator_flags)
                bar.update(1)
            with tqdm(total=step_count, desc="Running coupled RK4", unit="step") as bar:
                for start in range(0, step_count, progress_chunk_steps):
                    count = min(progress_chunk_steps, step_count - start)
                    y = _integrate_chunk(y, state.orbit.julian_date, self.dt,
                                         full_steps, remainder, runtime, output,
                                         start, count, record_every, sensor_every,
                                         sensor_config, clean, reading, seed,
                                         estimator_kind, estimate_q, estimate_w, estimate_p, acceleration_psd, estimates, estimator_flags)
                    bar.update(count)
        elif step_count:
            _integrate_chunk(y, state.orbit.julian_date, self.dt, full_steps,
                             remainder, runtime, output, 0, step_count, record_every,
                             sensor_every, sensor_config, clean, reading, seed,
                             estimator_kind, estimate_q, estimate_w, estimate_p, acceleration_psd, estimates, estimator_flags)
        return ((elapsed, output, clean, reading, estimates) if estimator_kind else
                (elapsed, output, clean, reading) if sensor_every else (elapsed, output))

    def step(self, state, satellite, environment, dynamics):
        if self._configuration != (id(satellite), id(environment), id(dynamics)):
            self._runtime = _prepare(satellite, environment, dynamics)
            self._configuration = (id(satellite), id(environment), id(dynamics))
        y = pack_state(state, self._runtime.wheel_commands.size)
        if self._runtime.wheel_commands.size:
            _project_wheel_momentum(y, self._runtime)
        result = _rk4_step(y, state.orbit.julian_date, self.dt, self._runtime)
        return unpack_state(result, state, self.dt)
