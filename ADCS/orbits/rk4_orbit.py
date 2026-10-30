"""Sequential, attitude-coupled RK4 orbit propagation."""

from __future__ import annotations

import numpy as np
from numba import njit

from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import EarthConstants, TimeConstants
from ADCS.state import State


@njit(cache=True)
def _earth_acceleration(r, mu, earth_radius, j2, zonal_J, j_coeffs):
    """Central, J2, and requested higher zonal acceleration [km/s²]."""
    radius2 = np.dot(r, r)
    radius = np.sqrt(radius2)
    radius3 = radius2 * radius
    acceleration = -mu * r / radius3

    if zonal_J >= 2:
        z2 = r[2] * r[2]
        factor = 1.5 * j2 * mu * earth_radius * earth_radius / radius**5
        common = 5.0 * z2 / radius2
        acceleration[0] += factor * r[0] * (common - 1.0)
        acceleration[1] += factor * r[1] * (common - 1.0)
        acceleration[2] += factor * r[2] * (common - 3.0)

    if zonal_J > 2:
        s = r[2] / radius
        p_previous_previous = 1.0
        p_previous = s
        dp_previous_previous = 0.0
        dp_previous = 1.0
        for n in range(2, zonal_J + 1):
            p_n = ((2 * n - 1) * s * p_previous - (n - 1) * p_previous_previous) / n
            dp_n = ((2 * n - 1) * (p_previous + s * dp_previous) - (n - 1) * dp_previous_previous) / n
            if n >= 3:
                jn = j_coeffs[n - 2]
                if jn != 0.0:
                    coefficient = -mu * jn * earth_radius**n
                    alpha = -(n + 1) * radius ** (-(n + 3)) * p_n - r[2] * radius ** (-(n + 4)) * dp_n
                    beta = radius ** (-(n + 2)) * dp_n
                    acceleration[0] += coefficient * alpha * r[0]
                    acceleration[1] += coefficient * alpha * r[1]
                    acceleration[2] += coefficient * (alpha * r[2] + beta)
            p_previous_previous = p_previous
            p_previous = p_n
            dp_previous_previous = dp_previous
            dp_previous = dp_n
    return acceleration


@njit(cache=True)
def _third_body_acceleration(r, body_position, mu):
    """Geocentric differential acceleration from a distant body [km/s²]."""
    relative = body_position - r
    relative_norm = np.sqrt(np.dot(relative, relative))
    body_norm = np.sqrt(np.dot(body_position, body_position))
    return mu * (relative / relative_norm**3 - body_position / body_norm**3)


@njit(cache=True)
def _rk4_combine(y0, k1, k2, k3, k4, dt):
    return y0 + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


@njit(cache=True)
def _body_force_acceleration(q, force_body, mass):
    """Rotate a body-frame force [N] to ECI acceleration [km/s²]."""
    q0, q1, q2, q3 = q
    f0, f1, f2 = force_body
    ax = (q0*q0 + q1*q1 - q2*q2 - q3*q3) * f0 + 2.0 * (q1*q2 - q0*q3) * f1 + 2.0 * (q1*q3 + q0*q2) * f2
    ay = 2.0 * (q1*q2 + q0*q3) * f0 + (q0*q0 - q1*q1 + q2*q2 - q3*q3) * f1 + 2.0 * (q2*q3 - q0*q1) * f2
    az = 2.0 * (q1*q3 - q0*q2) * f0 + 2.0 * (q2*q3 + q0*q1) * f1 + (q0*q0 - q1*q1 - q2*q2 + q3*q3) * f2
    return np.array([ax, ay, az]) / (mass * 1000.0)


class RK4Orbit(Orbit):
    """Orbit advanced alongside attitude using Earth, Sun, Moon, and spacecraft forces.

    The initial state is available immediately. ``advance`` adds one state at a
    time; future states cannot be queried until they have been propagated.
    Spacecraft disturbance forces (drag, SRP, thrust, etc.) must be attached to
    the ``Satellite`` passed to ``advance``. They are evaluated at each RK4
    stage using the interpolated attitude and the stage orbital state.

    ``zonal_J`` has the same meaning as on :class:`Orbit`; Earth central gravity
    is always included. Positions and velocities are ECI km and km/s.

    References
    ----------
    * Classical Runge–Kutta integration: J. C. Butcher, “Coefficients for the
      Study of Runge-Kutta Integration Processes,” *The Computer Journal*,
      6(1), 1963, pp. 50–54,
      `doi:10.1093/comjnl/6.1.50 <https://doi.org/10.1093/comjnl/6.1.50>`_.
    * Earth gravity coefficients: Lemoine et al.,
      `NASA/TP-1998-206861 (EGM96) <https://ntrs.nasa.gov/citations/19980218814>`_.
    * Sun and Moon gravitational parameters: `JPL Astrodynamic Parameters
      <https://ssd.jpl.nasa.gov/astro_par.html>`_.
    * Drag model: see :class:`ADCS.satellite_hardware.disturbances.Drag_Disturbance`
      and the corresponding entry in the package citation guide.
    """

    coupled_to_attitude = True

    def __init__(self, os0: Orbital_State, *, zonal_J: int = 2, environment=None):
        super().__init__(os0=os0, zonal_J=zonal_J, environment=environment, verbose=False)
        self._third_body_cache = {}

    def _third_body_positions(self, j2000: float):
        key = float(j2000)
        cached = self._third_body_cache.get(key)
        if cached is not None:
            return cached
        ephemeris = self.environment.ephemeris
        positions = []
        for name in ("sun", "moon"):
            try:
                positions.append(np.asarray(
                    ephemeris.position_eci(name, key, apparent=False), dtype=float
                ).reshape(3))
            except (KeyError, ValueError, NotImplementedError) as exc:
                raise ValueError(f"RK4Orbit requires an ephemeris with {name} positions") from exc
        cached = tuple(positions)
        self._third_body_cache[key] = cached
        # Predictor-corrector calls reuse the same stage epochs. Keep this
        # bounded so long simulations do not retain an unbounded cache.
        if len(self._third_body_cache) > 16:
            oldest = next(iter(self._third_body_cache))
            del self._third_body_cache[oldest]
        return cached

    def _stage_state(self, j2000: float, r: np.ndarray, v: np.ndarray) -> Orbital_State:
        state = Orbital_State(J2000=j2000, R=r, V=v, environment=self.environment)
        state._skip_jacobians = True
        return state

    def _derivative(self, t: float, y: np.ndarray, attitude: State, satellite, start: Orbital_State, body_positions=None):
        j2000 = start.J2000 + t * TimeConstants.sec2cent
        r, v = y[:3], y[3:]
        os = self._stage_state(j2000, r, v)
        acceleration = _earth_acceleration(
            r, start.mu_e, start.R_e, start.J2coeff, self._zonal_J, start.Jcoeffs
        )
        if body_positions is None:
            body_positions = self._third_body_positions(j2000)
        acceleration = acceleration + _third_body_acceleration(r, body_positions[0], EarthConstants.mu_sun)
        acceleration = acceleration + _third_body_acceleration(r, body_positions[1], EarthConstants.mu_moon)

        force_body = satellite.dist_forces(x=attitude, os=os)
        acceleration = acceleration + _body_force_acceleration(attitude.q, force_body, float(satellite.mass))
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
        t0 = start.J2000
        tmid = t0 + dt * 0.5 * TimeConstants.sec2cent
        t1 = t0 + dt * TimeConstants.sec2cent
        body0 = self._third_body_positions(t0)
        bodymid = self._third_body_positions(tmid)
        body1 = self._third_body_positions(t1)
        k1 = self._derivative(0.0, y0, x0, satellite, start, body0)
        k2 = self._derivative(dt / 2, y0 + dt * k1 / 2, midpoint, satellite, start, bodymid)
        k3 = self._derivative(dt / 2, y0 + dt * k2 / 2, midpoint, satellite, start, bodymid)
        k4 = self._derivative(dt, y0 + dt * k3, x1, satellite, start, body1)
        y1 = _rk4_combine(y0, k1, k2, k3, k4, dt)
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
