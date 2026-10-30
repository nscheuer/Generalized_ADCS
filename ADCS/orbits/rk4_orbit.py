"""Sequential, attitude-coupled RK4 orbit propagation."""

from __future__ import annotations

import numpy as np

from ADCS.helpers.math_helpers import rot_mat
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import EarthConstants, TimeConstants
from ADCS.state import State


class RK4Orbit(Orbit):
    """Orbit advanced alongside attitude using Earth, Sun, Moon, and spacecraft forces.

    The initial state is available immediately. ``advance`` adds one state at a
    time; future states cannot be queried until they have been propagated.
    Spacecraft disturbance forces (drag, SRP, thrust, etc.) must be attached to
    the ``Satellite`` passed to ``advance``. They are evaluated at each RK4
    stage using the interpolated attitude and the stage orbital state.

    ``zonal_J`` has the same meaning as on :class:`Orbit`; Earth central gravity
    is always included. Positions and velocities are ECI km and km/s.
    """

    def __init__(self, os0: Orbital_State, *, zonal_J: int = 2, environment=None):
        super().__init__(os0=os0, zonal_J=zonal_J, environment=environment, verbose=False)

    @staticmethod
    def _third_body_acceleration(r: np.ndarray, body_position: np.ndarray, mu: float) -> np.ndarray:
        """Geocentric differential acceleration from a distant body [km/s²]."""
        relative = body_position - r
        return mu * (
            relative / np.linalg.norm(relative) ** 3
            - body_position / np.linalg.norm(body_position) ** 3
        )

    def _stage_state(self, j2000: float, r: np.ndarray, v: np.ndarray) -> Orbital_State:
        state = Orbital_State(J2000=j2000, R=r, V=v, environment=self.environment)
        state._skip_jacobians = True
        return state

    def _derivative(self, t: float, y: np.ndarray, attitude: State, satellite, start: Orbital_State):
        j2000 = start.J2000 + t * TimeConstants.sec2cent
        r, v = y[:3], y[3:]
        os = self._stage_state(j2000, r, v)
        _, acceleration = Orbital_State._orbit_dynamics_raw(
            r, v, start.mu_e, start.R_e, start.J2coeff,
            self._zonal_J, Jcoeffs=start.Jcoeffs,
        )
        ephemeris = self.environment.ephemeris
        for name, mu in (("sun", EarthConstants.mu_sun), ("moon", EarthConstants.mu_moon)):
            try:
                body = np.asarray(
                    ephemeris.position_eci(name, j2000, apparent=False), dtype=float
                ).reshape(3)
            except (KeyError, ValueError, NotImplementedError) as exc:
                raise ValueError(f"RK4Orbit requires an ephemeris with {name} positions") from exc
            acceleration = acceleration + self._third_body_acceleration(r, body, mu)

        force_body = satellite.dist_forces(x=attitude, os=os)
        acceleration = acceleration + rot_mat(attitude.q) @ force_body / (float(satellite.mass) * 1000.0)
        return np.concatenate((v, acceleration))

    def predict(self, x0: State, x1: State, satellite, dt: float) -> Orbital_State:
        """Calculate one RK4 step without changing the stored orbit."""
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if not np.isfinite(satellite.mass) or satellite.mass <= 0:
            raise ValueError("satellite.mass must be finite and positive")
        start = self.states[self.times[-1]]
        y0 = np.concatenate((start.R, start.V))
        midpoint = x0.interpolate(x1, 0.5)
        k1 = self._derivative(0.0, y0, x0, satellite, start)
        k2 = self._derivative(dt / 2, y0 + dt * k1 / 2, midpoint, satellite, start)
        k3 = self._derivative(dt / 2, y0 + dt * k2 / 2, midpoint, satellite, start)
        k4 = self._derivative(dt, y0 + dt * k3, x1, satellite, start)
        y1 = y0 + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return self._stage_state(start.J2000 + dt * TimeConstants.sec2cent, y1[:3], y1[3:])

    def advance(self, x0: State, x1: State, satellite, dt: float) -> Orbital_State:
        """Propagate and store the next state after both attitude endpoints are known."""
        return self._commit(self.predict(x0, x1, satellite, dt))

    def _commit(self, next_state: Orbital_State) -> Orbital_State:
        """Store a state produced by ``predict`` after attitude correction."""
        if next_state.J2000 <= self.max_time():
            raise ValueError("next_state must be later than the current orbit state")
        if next_state.environment is not self.environment:
            raise ValueError("next_state must use this orbit's environment")
        next_state = next_state.copy()
        self.states[next_state.J2000] = next_state
        self.times = np.append(self.times, next_state.J2000)
        return next_state.copy()

    def get_os(self, J2000: float) -> Orbital_State:
        """Return a stored state; coupled future states require ``advance``."""
        t = float(J2000)
        matches = np.flatnonzero(np.isclose(self.times, t, rtol=0.0, atol=1e-2 / TimeConstants.cent2sec))
        if matches.size:
            index = matches[np.argmin(np.abs(self.times[matches] - t))]
            return self.states[self.times[index]].copy()
        raise ValueError("RK4Orbit has no state at this time; advance it sequentially first")
