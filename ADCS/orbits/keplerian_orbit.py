"""Fast two-body orbit propagation from osculating Keplerian elements."""

__all__ = ["KeplerianOrbit"]

import numpy as np
from tqdm import tqdm

from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants


def _solve_kepler(M, e, tol=1e-12, max_iter=50):
    """Eccentric anomaly from mean anomaly by Newton's method; raises if it does not converge."""
    M = np.mod(M, 2.0 * np.pi)
    E = np.pi if e > 0.8 else M  # E = M stalls just past periapsis for high e
    for _ in range(max_iter):
        step = (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
        E -= step
        if abs(step) < tol:
            return E
    raise RuntimeError(f"Kepler's equation did not converge for M={M}, e={e}")


class KeplerianOrbit(Orbit):
    r"""Attitude-independent elliptic two-body orbit.

    The initial osculating elements are held constant and mean anomaly advances
    at the Keplerian mean motion. For semi-major axis :math:`a`, eccentricity
    :math:`e`, gravitational parameter :math:`\mu`, and elapsed time :math:`t`,

    .. math::

       n = \sqrt{\frac{\mu}{a^3}}, \qquad
       M(t) = M_0 + nt, \qquad E - e\sin E = M(t).

    :param os0: Initial state on a bound elliptic orbit.
    :type os0: ADCS.orbits.orbital_state.Orbital_State
    :param end_time: Final epoch in J2000 centuries; defaults to ``os0.J2000``.
    :type end_time: float or None
    :param dt: Requested node spacing in seconds for the precomputed trajectory.
    :type dt: float or None
    :param fast: Backward-compatible argument passed to :class:`Orbit`.
    :type fast: bool
    :param verbose: Show a tqdm progress bar while propagating the trajectory.
    :type verbose: bool
    :param zonal_J: Gravity setting passed to :class:`Orbit` (not applied by this model).
    :type zonal_J: int
    :param magnetic_model: Optional magnetic-field model override.
    :type magnetic_model: str or None
    :param environment: Environment used for returned orbital states.
    :type environment: ADCS.environment.environment.Environment or None
    :raises TypeError: If ``os0`` is not one orbital state.
    :raises ValueError: If ``os0`` is not bound and elliptic.
    """

    def __init__(self, os0, end_time=None, dt=None, fast=True, verbose=True,
                 zonal_J=0, magnetic_model=None, environment=None):
        if not isinstance(os0, Orbital_State):
            raise TypeError("KeplerianOrbit requires one initial Orbital_State")
        if not np.isfinite(os0.eccentricity) or os0.eccentricity >= 1.0:
            raise ValueError("KeplerianOrbit requires a bound elliptic initial orbit")
        self._a, self._e, inc, raan, argp, nu0 = os0.to_keplerian(mu=os0.mu_e)
        self._inc, self._raan, self._argp, nu0 = np.deg2rad([inc, raan, argp, nu0])
        self._mu = float(os0.mu_e)
        self._mean_motion = np.sqrt(self._mu / self._a**3)
        E0 = 2.0 * np.arctan2(np.sqrt(1.0 - self._e) * np.sin(nu0 / 2.0),
                              np.sqrt(1.0 + self._e) * np.cos(nu0 / 2.0))
        self._mean_anomaly0 = E0 - self._e * np.sin(E0)
        super().__init__(os0, end_time=end_time, dt=dt, fast=fast, verbose=verbose,
                         zonal_J=zonal_J, magnetic_model=magnetic_model,
                         environment=environment)

    def _propagate_trajectory(self, os0, times_arr, mu_e, R_e, J2coeff, Jcoeffs, *, verbose=True):
        N = len(times_arr)
        R_hist = np.empty((N, 3), dtype=float)
        V_hist = np.empty((N, 3), dtype=float)
        R_hist[0] = os0.R
        V_hist[0] = os0.V
        p = self._a * (1.0 - self._e**2)
        sqrt_mu_p = np.sqrt(self._mu / p)

        co, so = np.cos(self._raan), np.sin(self._raan)
        ci, si = np.cos(self._inc), np.sin(self._inc)
        cw, sw = np.cos(self._argp), np.sin(self._argp)
        rotation = np.array([
            [co * cw - so * sw * ci, -co * sw - so * cw * ci, so * si],
            [so * cw + co * sw * ci, -so * sw + co * cw * ci, -co * si],
            [sw * si, cw * si, ci],
        ])

        for j in tqdm(range(1, N), desc="Propagating Keplerian Orbit", unit="step", disable=not verbose):
            elapsed = (times_arr[j] - times_arr[0]) * TimeConstants.cent2sec
            M = self._mean_anomaly0 + self._mean_motion * elapsed
            E = _solve_kepler(M, self._e)
            nu = 2.0 * np.arctan2(np.sqrt(1.0 + self._e) * np.sin(E / 2.0),
                                  np.sqrt(1.0 - self._e) * np.cos(E / 2.0))
            radius = p / (1.0 + self._e * np.cos(nu))
            r_pf = radius * np.array([np.cos(nu), np.sin(nu), 0.0])
            v_pf = sqrt_mu_p * np.array([-np.sin(nu), self._e + np.cos(nu), 0.0])
            R_hist[j] = rotation @ r_pf
            V_hist[j] = rotation @ v_pf
        return R_hist, V_hist
