"""Fixed-step, attitude-only RK4 propagation with a frozen orbital state."""

import math

import numpy as np
from numba import njit

from GenADCS.backends.python.config import DynamicsConfig
from GenADCS.backends.python.environment import Environment
from GenADCS.backends.python.hardware import Satellite
from GenADCS.backends.python.helpers.math.derivatives import quaternion_derivative
from GenADCS.backends.python.propagation.coupled_propagators._coupled_dynamics import (
    SECONDS_PER_DAY,
    RuntimeParameters,
    _body_torque,
    _environment_conditions,
    _internal_derivative,
    prepare_runtime,
    project_wheel_momentum,
)
from GenADCS.backends.python.propagation.coupled_propagators._state_layout import (
    pack_state,
)
from GenADCS.backends.python.state import AttitudeState, State, WheelMomentumState


@njit(cache=True)
def _attitude_derivative(
    y: np.ndarray, epoch_jd: float, p: RuntimeParameters
) -> np.ndarray:
    q = y[:4].copy()
    q /= np.sqrt(np.dot(q, q))
    omega = y[4:7]
    position = y[7:10]
    velocity = y[10:13]
    conditions = _environment_conditions(position, velocity, epoch_jd, p)
    torque = _body_torque(position, q, p, conditions)
    angular_momentum = p.inertia @ omega
    if p.wheel_axes.shape[0] == 0:
        omega_dot = p.inverse_inertia @ (torque - np.cross(omega, angular_momentum))
    else:
        for wheel in range(p.wheel_axes.shape[0]):
            angular_momentum += p.wheel_axes[wheel] * y[13 + wheel]
        omega_dot = p.inverse_bus_inertia @ (torque - np.cross(omega, angular_momentum))
    dy = np.zeros(y.size)
    dy[:4] = quaternion_derivative(q, omega)
    dy[4:7] = omega_dot
    if p.wheel_axes.shape[0] != 0:
        _internal_derivative(
            dy, y, q, omega, position, velocity, epoch_jd, conditions, omega_dot, p
        )
    return dy


@njit(cache=True)
def _rk4_attitude_step(
    y: np.ndarray, epoch_jd: float, dt: float, p: RuntimeParameters
) -> np.ndarray:
    k1 = _attitude_derivative(y, epoch_jd, p)
    k2 = _attitude_derivative(
        y + 0.5 * dt * k1, epoch_jd + 0.5 * dt / SECONDS_PER_DAY, p
    )
    k3 = _attitude_derivative(
        y + 0.5 * dt * k2, epoch_jd + 0.5 * dt / SECONDS_PER_DAY, p
    )
    k4 = _attitude_derivative(y + dt * k3, epoch_jd + dt / SECONDS_PER_DAY, p)
    result = y + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    result[:4] /= np.sqrt(np.dot(result[:4], result[:4]))
    if p.wheel_commands.size:
        project_wheel_momentum(result, p)
    return result


@njit(cache=True)
def _integrate_attitude(
    y: np.ndarray,
    initial_jd: float,
    dt: float,
    full_steps: int,
    remainder: float,
    p: RuntimeParameters,
    record_every: int,
    output: np.ndarray,
) -> None:
    """Run every attitude step inside one Numba call."""
    total_steps = full_steps + int(remainder > 0.0)
    for row in range(total_steps):
        elapsed = row * dt
        step_dt = remainder if row == full_steps else dt
        y = _rk4_attitude_step(y, initial_jd + elapsed / SECONDS_PER_DAY, step_dt, p)
        completed = row + 1
        if completed % record_every == 0:
            output[completed // record_every] = y
        elif completed == total_steps:
            output[-1] = y


class RK4AttitudePropagator:
    def __init__(self, dt: float) -> None:
        self.dt = float(dt)
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive seconds")

    def integrate(
        self,
        state: State,
        satellite: Satellite,
        environment: Environment,
        dynamics: DynamicsConfig,
        tf: float,
        record_every: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return elapsed seconds and packed states; all steps run in Numba.

        Orbit position and velocity remain fixed for this attitude-only run.
        """
        tf = float(tf)
        if not math.isfinite(tf) or tf < 0.0:
            raise ValueError("tf must be finite and nonnegative seconds")
        if (
            isinstance(record_every, bool)
            or not isinstance(record_every, int)
            or record_every < 1
        ):
            raise ValueError("record_every must be a positive integer")
        runtime = prepare_runtime(satellite, environment, dynamics)
        y = pack_state(state, runtime.wheel_commands.size)
        if runtime.wheel_commands.size:
            project_wheel_momentum(y, runtime)
        full_steps = math.floor(tf / self.dt)
        remainder = tf - full_steps * self.dt
        if remainder <= 1e-12 * max(1.0, tf):
            remainder = 0.0
        total_steps = full_steps + int(remainder > 0.0)
        sampled_steps = np.arange(0, total_steps + 1, record_every, dtype=np.int64)
        elapsed = np.minimum(sampled_steps.astype(np.float64) * self.dt, tf)
        if sampled_steps[-1] != total_steps:
            elapsed = np.append(elapsed, tf)
        else:
            elapsed[-1] = tf
        output = np.empty((elapsed.size, y.size), dtype=np.float64)
        output[0] = y
        if total_steps:
            _integrate_attitude(
                y,
                state.orbit.julian_date,
                self.dt,
                full_steps,
                remainder,
                runtime,
                record_every,
                output,
            )
        return elapsed, output

    def step(
        self,
        state: State,
        satellite: Satellite,
        environment: Environment,
        dynamics: DynamicsConfig,
    ) -> State:
        runtime = prepare_runtime(satellite, environment, dynamics)
        y = pack_state(state, runtime.wheel_commands.size)
        if runtime.wheel_commands.size:
            project_wheel_momentum(y, runtime)
        result = _rk4_attitude_step(y, state.orbit.julian_date, self.dt, runtime)
        return State(
            AttitudeState(result[:4].copy(), result[4:7].copy()),
            state.orbit,
            WheelMomentumState(wheel_momentum=result[13:].copy()),
        )
