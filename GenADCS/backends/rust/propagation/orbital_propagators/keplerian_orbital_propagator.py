"""Two-body Keplerian propagator backed by the Rust propagation crate."""

import math

import numpy as np

from GenADCS.backends.rust.propagation.coupled_propagators._coupled import (
    integrate_keplerian,
    keplerian_step,
)
from GenADCS.backends.rust.propagation.coupled_propagators._state_layout import pack_state
from GenADCS.backends.rust.state import OrbitalState, State


class KeplerianOrbitalPropagator:
    """Propagate a Rust-backed state under central gravity."""

    def __init__(self, dt: float) -> None:
        self.dt = float(dt)
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive seconds")

    @staticmethod
    def _mu(environment) -> float:
        mu = float(environment.planetary_body.gravitational_parameter)
        if not math.isfinite(mu) or mu <= 0.0:
            raise ValueError("gravitational parameter must be finite and positive")
        return mu

    def integrate(
        self,
        state: State,
        environment,
        dynamics,
        tf: float,
        progress: bool = False,
        progress_chunk_steps: int = 500,
        record_every: int = 1,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        del dynamics, progress, progress_chunk_steps, kwargs
        tf = float(tf)
        if not math.isfinite(tf) or tf < 0.0:
            raise ValueError("tf must be finite and nonnegative seconds")
        if isinstance(record_every, bool) or not isinstance(record_every, int) or record_every < 1:
            raise ValueError("record_every must be a positive integer")
        wheels = np.asarray(state.internal.wheel_momentum).size
        y = pack_state(state, wheels).tolist()
        times, rows = integrate_keplerian(y, self._mu(environment), self.dt, tf, record_every)
        return np.asarray(times, dtype=np.float64), np.asarray(rows, dtype=np.float64)

    def step(self, state: State, environment, dynamics) -> State:
        del dynamics
        position, velocity = keplerian_step(
            np.asarray(state.orbit.position, dtype=np.float64).tolist(),
            np.asarray(state.orbit.velocity, dtype=np.float64).tolist(),
            self._mu(environment),
            self.dt,
        )
        orbit = OrbitalState.from_cartesian(
            np.asarray(position),
            np.asarray(velocity),
            state.orbit.julian_date + self.dt / 86400.0,
            state.orbit.ephemeris,
        )
        return State(state.attitude, orbit, state.internal)
