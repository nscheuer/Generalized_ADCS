r"""SGP4 propagation from a TLE fitted to the initial state.

The initial osculating state is turned into SGP4 mean elements by a short
fixed-point iteration, so that SGP4 evaluated at the epoch reproduces the
initial position to within the precision of the TLE format (tens of metres).
This is a fit at one instant, not an orbit determination over an arc.
"""

__all__ = ["SGP4Orbit"]

import warnings

import numpy as np
from skyfield.api import EarthSatellite
from tqdm import tqdm

from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State, _classical_from_rv, _tle_lines
from ADCS.satellite_hardware.disturbances.drag_disturbance import Drag_Disturbance


def _average_bstar(satellite):
    r"""Estimate B* using an isotropic attitude average of drag face areas.

    For a unit direction :math:`\hat{u}` uniformly distributed over a sphere,

    .. math::

       \left\langle \max(0, \hat{n}\cdot\hat{u}) \right\rangle = \frac{1}{4}.

    Thus the average drag area is :math:`\overline{C_D A} =
    \frac{1}{4}\sum_i C_{D,i} A_i`. The standard SGP4/TLE ballistic term is
    estimated as :math:`B^* = \frac{1}{2}\rho_0\overline{C_D A}/m`, using
    :math:`\rho_0=2.461\times10^{-5}` in the SGP4 reference-density convention.

    :param satellite: Satellite with positive mass and a ``Drag_Disturbance``.
        Without a drag disturbance (or without a satellite) B* is 0 and a
        warning is issued, since SGP4 then models no decay.
    :type satellite: ADCS.satellite_hardware.satellite.satellite.Satellite or None
    :return: Estimated B* in the standard TLE convention.
    :rtype: float
    :raises ValueError: If the satellite has a drag disturbance but no positive mass.
    """
    if satellite is None:
        return 0.0
    drag = next((d for d in satellite.disturbances if isinstance(d, Drag_Disturbance)), None)
    if drag is None:
        warnings.warn("SGP4Orbit: the satellite has no Drag_Disturbance, so B* is set to 0 (no drag decay). "
                      "Pass bstar= to set it explicitly.", stacklevel=3)
        return 0.0
    if satellite.mass <= 0:
        raise ValueError("satellite mass must be positive to derive B*")
    # Isotropic attitude average of projected area: integral(max(0,n·u)) dΩ / 4π = 1/4.
    average_cd_area = float(np.sum(drag.CDs * drag.areas)) / 4.0
    # SGP4's reference density is 2.461e-5 kg/m², yielding standard TLE B* units.
    return 0.5 * 2.461e-5 * average_cd_area / satellite.mass


def _equinoctial(a, e, inc, raan, argp, mean_anomaly):
    """Classical (radians) -> equinoctial ``(a, f, g, h, k, L)``; nonsingular for circular and equatorial orbits."""
    lon_peri = raan + argp
    t = np.tan(inc / 2.0)
    return np.array([a, e * np.cos(lon_peri), e * np.sin(lon_peri), t * np.cos(raan), t * np.sin(raan),
                     lon_peri + mean_anomaly], dtype=float)


def _classical(q):
    """Equinoctial ``(a, f, g, h, k, L)`` -> classical ``(a, e, inc, raan, argp, mean_anomaly)`` in radians."""
    a, f, g, h, k, L = q
    e = float(np.hypot(f, g))
    inc = 2.0 * np.arctan(np.hypot(h, k))
    raan = np.arctan2(k, h) if np.hypot(h, k) > 1e-12 else 0.0
    lon_peri = np.arctan2(g, f) if e > 1e-12 else raan
    argp = lon_peri - raan
    return a, e, inc, raan % (2 * np.pi), argp % (2 * np.pi), (L - lon_peri) % (2 * np.pi)


class SGP4Orbit(Orbit):
    r"""Attitude-independent SGP4 orbit propagated from an initial state.

    SGP4 mean elements are fitted to ``os0`` so that the SGP4 position at the
    epoch matches ``os0.R``. The fit is a fixed-point iteration on
    equinoctial elements: each pass propagates the current TLE to the epoch,
    takes the osculating elements of the result, and corrects the mean
    elements by the difference to the target. The TLE format limits the
    agreement to a few tens of metres (four-decimal angles, 1e-7 eccentricity).
    After the fit SGP4 advances the satellite independently of the attitude loop.

    If ``bstar`` is omitted and a satellite is provided, B* is estimated from
    its isotropic average drag area. For faces :math:`i` with area :math:`A_i`
    and drag coefficient :math:`C_{D,i}`,

    .. math::

       \overline{C_D A} = \frac{1}{4}\sum_i C_{D,i}A_i, \qquad
       B^* = \frac{1}{2}\rho_0\frac{\overline{C_D A}}{m}.

    Here :math:`m` is satellite mass and :math:`\rho_0=2.461\times10^{-5}`
    is the SGP4 reference-density convention. This is a rough attitude-average
    estimate; a manually supplied B* takes precedence. Without a drag
    disturbance B* is 0 and a warning is issued. The SGP4/TLE B* field
    convention is inverse Earth radii.

    :param os0: Initial orbital state; it must define a bound elliptic orbit.
    :type os0: ADCS.orbits.orbital_state.Orbital_State
    :param end_time: Final epoch in J2000 centuries; defaults to ``os0.J2000``.
    :type end_time: float or None
    :param dt: Requested node spacing in seconds for the precomputed trajectory.
    :type dt: float or None
    :param fast: Backward-compatible argument passed to :class:`Orbit`.
    :type fast: bool
    :param verbose: Show a tqdm progress bar while precomputing the trajectory.
    :type verbose: bool
    :param zonal_J: Backward-compatible gravity setting passed to :class:`Orbit`.
    :type zonal_J: int
    :param magnetic_model: Optional magnetic-field model override.
    :type magnetic_model: str or None
    :param environment: Environment used to construct returned orbital states.
    :type environment: ADCS.environment.environment.Environment or None
    :param satellite: Satellite geometry used to estimate B* when ``bstar`` is
        omitted.
    :type satellite: ADCS.satellite_hardware.satellite.satellite.Satellite or None
    :param bstar: Manual SGP4/TLE B* value, in inverse Earth radii. If supplied,
        this value overrides the geometry-based estimate.
    :type bstar: float or None
    :param fit_iterations: Maximum fixed-point passes of the mean-element fit.
    :type fit_iterations: int
    :param fit_tol_km: Position residual at the epoch below which the fit stops.
    :type fit_tol_km: float
    :raises TypeError: If ``os0`` is not one :class:`Orbital_State`.
    :raises ValueError: If B* is not finite, geometry-derived B* is invalid, or
        ``os0`` is not a bound elliptic orbit.

    .. attribute:: tle

       The fitted ``(line1, line2)``.

    .. attribute:: fit_residual_km

       Distance between SGP4 at the epoch and ``os0.R`` after the fit.
    """

    def __init__(self, os0, end_time=None, dt=None, fast=True, verbose=True,
                 zonal_J=0, magnetic_model=None, environment=None,
                 satellite=None, bstar=None, fit_iterations=20, fit_tol_km=0.02):
        if not isinstance(os0, Orbital_State):
            raise TypeError("SGP4Orbit requires one initial Orbital_State")
        if bstar is None:
            bstar = _average_bstar(satellite)
        self.bstar = float(bstar)
        if not np.isfinite(self.bstar):
            raise ValueError("bstar must be finite")
        self.tle, self.fit_residual_km, self.fit_passes = self._fit_tle(os0, int(fit_iterations), float(fit_tol_km))
        self._tle_lines = self.tle
        super().__init__(os0, end_time=end_time, dt=dt, fast=fast, verbose=verbose,
                         zonal_J=zonal_J, magnetic_model=magnetic_model,
                         environment=environment)

    def _lines_from_mean(self, q, epoch_dt, mu):
        a, e, inc, raan, argp, ma = _classical(q)
        if not (np.isfinite(a) and a > 0 and 0.0 <= e < 1.0):
            raise ValueError("SGP4 mean-element fit left the elliptic range; the orbit is not representable by a TLE")
        mean_motion = np.sqrt(mu / a**3) * 86400.0 / (2.0 * np.pi)
        return _tle_lines(epoch_dt, inc, raan, e, argp, ma, mean_motion, bstar=self.bstar)

    def _fit_tle(self, os0, max_passes, tol_km):
        """Fixed-point fit of SGP4 mean elements to the osculating state at the epoch."""
        ts = os0.ts
        mu = float(os0.mu_e)
        t_epoch = ts.tt_jd(float(os0.J2000) * 36525.0 + 2451545.0)
        epoch_dt = os0.sf_pos.t.utc_datetime()
        a, e, inc, raan, argp, nu = _classical_from_rv(os0.R, os0.V, mu)
        E = 2.0 * np.arctan2(np.sqrt(1.0 - e) * np.sin(nu / 2.0), np.sqrt(1.0 + e) * np.cos(nu / 2.0))
        target = _equinoctial(a, e, inc, raan, argp, (E - e * np.sin(E)) % (2 * np.pi))
        mean = target.copy()
        best = (None, np.inf, 0)
        for k in range(1, max_passes + 1):
            lines = self._lines_from_mean(mean, epoch_dt, mu)
            g = EarthSatellite(*lines, ts=ts).at(t_epoch)
            r = np.asarray(g.position.km, dtype=float).reshape(3)
            v = np.asarray(g.velocity.km_per_s, dtype=float).reshape(3)
            residual = float(np.linalg.norm(r - os0.R))
            if residual < best[1]:
                best = (lines, residual, k)
            if residual < tol_km:
                break
            a_, e_, i_, O_, w_, nu_ = _classical_from_rv(r, v, mu)
            E_ = 2.0 * np.arctan2(np.sqrt(1.0 - e_) * np.sin(nu_ / 2.0), np.sqrt(1.0 + e_) * np.cos(nu_ / 2.0))
            osc = _equinoctial(a_, e_, i_, O_, w_, (E_ - e_ * np.sin(E_)) % (2 * np.pi))
            delta = target - osc
            delta[5] = (delta[5] + np.pi) % (2 * np.pi) - np.pi
            mean = mean + delta
        return best

    def _propagate_trajectory(self, os0, times_arr, mu_e, R_e, J2coeff, Jcoeffs, *, verbose=True):
        """Propagate one requested epoch at a time with the fitted SGP4 TLE."""
        self._satellite = EarthSatellite(*self._tle_lines, ts=self.environment.ephemeris.ts)
        N = len(times_arr)
        R_hist = np.empty((N, 3), dtype=float)
        V_hist = np.empty((N, 3), dtype=float)
        R_hist[0] = os0.R
        V_hist[0] = os0.V
        ts = self.environment.ephemeris.ts
        for j in tqdm(range(1, N), desc="Propagating SGP4 Orbit", unit="step", disable=not verbose):
            sf_time = ts.tt_jd(float(times_arr[j]) * 36525.0 + 2451545.0)
            position = self._satellite.at(sf_time)
            R_hist[j] = np.asarray(position.position.km, dtype=float).reshape(3)
            V_hist[j] = np.asarray(position.velocity.km_per_s, dtype=float).reshape(3)
        return R_hist, V_hist
