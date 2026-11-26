"""Keplerian two-body propagation using a Numba universal-variable kernel."""

import math

import numpy as np
from numba import njit

from GenADCS.backends.python.config import DynamicsConfig
from GenADCS.backends.python.environment import Environment
from GenADCS.backends.python.propagation.coupled_propagators._state_layout import (
    pack_state,
)
from GenADCS.backends.python.state import OrbitalState, State


@njit(cache=True)
def _stumpff(z: float) -> tuple[float, float]:
    if z > 1e-8:
        root = math.sqrt(z)
        return (1.0 - math.cos(root)) / z, (root - math.sin(root)) / root**3
    if z < -1e-8:
        root = math.sqrt(-z)
        return (math.cosh(root) - 1.0) / -z, (math.sinh(root) - root) / root**3
    return 0.5 - z / 24.0 + z * z / 720.0, 1.0 / 6.0 - z / 120.0 + z * z / 5040.0


@njit(cache=True)
def keplerian_step(
    position: np.ndarray, velocity: np.ndarray, mu: float, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return two-body position and velocity after ``dt`` seconds, in SI units."""
    radius = np.linalg.norm(position)
    if radius <= 0.0:
        raise ValueError("orbital position must have positive radius")
    root_mu = math.sqrt(mu)
    radial_speed = np.dot(position, velocity) / radius
    alpha = 2.0 / radius - np.dot(velocity, velocity) / mu
    chi = root_mu * dt * (abs(alpha) if abs(alpha) > 1e-12 else 1.0 / radius)
    for _ in range(100):
        z = alpha * chi * chi
        c, s = _stumpff(z)
        residual = (
            radius * radial_speed / root_mu * chi * chi * c
            + (1.0 - alpha * radius) * chi**3 * s
            + radius * chi
            - root_mu * dt
        )
        derivative = (
            radius * radial_speed / root_mu * chi * (1.0 - z * s)
            + (1.0 - alpha * radius) * chi * chi * c
            + radius
        )
        change = residual / derivative
        chi -= change
        if abs(change) <= 1e-12 * max(1.0, abs(chi)):
            break
    else:
        raise RuntimeError("Keplerian universal anomaly did not converge")

    z = alpha * chi * chi
    c, s = _stumpff(z)
    f = 1.0 - chi * chi * c / radius
    g = dt - chi**3 * s / root_mu
    next_position = f * position + g * velocity
    next_radius = np.linalg.norm(next_position)
    f_dot = root_mu / (next_radius * radius) * (z * s - 1.0) * chi
    g_dot = 1.0 - chi * chi * c / next_radius
    return next_position, f_dot * position + g_dot * velocity


@njit(cache=True)
def _integrate_keplerian(
    y: np.ndarray,
    mu: float,
    dt: float,
    full_steps: int,
    remainder: float,
    record_every: int,
    output: np.ndarray,
) -> None:
    """Run all two-body steps and record samples inside one Numba call."""
    total_steps = full_steps + int(remainder > 0.0)
    for row in range(total_steps):
        step_dt = remainder if row == full_steps else dt
        position, velocity = keplerian_step(y[7:10], y[10:13], mu, step_dt)
        y[7:10] = position
        y[10:13] = velocity
        completed = row + 1
        if completed % record_every == 0:
            output[completed // record_every] = y
        elif completed == total_steps:
            output[-1] = y


class KeplerianOrbitalPropagator:
    def __init__(self, dt: float) -> None:
        self.dt = float(dt)
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive seconds")

    def integrate(
        self,
        state: State,
        environment: Environment,
        dynamics: DynamicsConfig,
        tf: float,
        record_every: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return elapsed seconds and packed states; every step runs in Numba."""
        tf = float(tf)
        if not math.isfinite(tf) or tf < 0.0:
            raise ValueError("tf must be finite and nonnegative seconds")
        if (
            isinstance(record_every, bool)
            or not isinstance(record_every, int)
            or record_every < 1
        ):
            raise ValueError("record_every must be a positive integer")
        mu = float(environment.planetary_body.gravitational_parameter)
        if not math.isfinite(mu) or mu <= 0.0:
            raise ValueError("gravitational parameter must be finite and positive")
        y = pack_state(state, state.internal.wheel_momentum.size)
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
            _integrate_keplerian(
                y, mu, self.dt, full_steps, remainder, record_every, output
            )
        return elapsed, output

    def step(
        self, state: State, environment: Environment, dynamics: DynamicsConfig
    ) -> State:
        """Advance the orbit while carrying attitude and wheel state forward."""
        mu = float(environment.planetary_body.gravitational_parameter)
        if not math.isfinite(mu) or mu <= 0.0:
            raise ValueError("gravitational parameter must be finite and positive")
        position, velocity = keplerian_step(
            state.orbit.position, state.orbit.velocity, mu, self.dt
        )
        orbit = OrbitalState.from_cartesian(
            position,
            velocity,
            state.orbit.julian_date + self.dt / 86400.0,
            state.orbit.ephemeris,
        )
        return State(state.attitude, orbit, state.internal)
