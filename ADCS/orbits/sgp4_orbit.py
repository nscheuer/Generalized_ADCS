r"""SGP4 propagation from an approximate TLE generated at the initial state.

The initial osculating state is converted to a TLE and propagated using
Skyfield's SGP4 implementation. TLE mean elements are fitted parameters, so
this conversion is a convenient initialization, not an orbit determination fit.
"""

__all__ = ["SGP4Orbit"]

import numpy as np
from skyfield.api import EarthSatellite
from tqdm import tqdm

from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.disturbances.drag_disturbance import Drag_Disturbance


def _tle_checksum(line):
    return line[:68] + str(sum(int(c) if c.isdigit() else (1 if c == "-" else 0) for c in line[:68]) % 10)


def _format_bstar(value):
    """Format B* in the implied-decimal/exponent TLE field."""
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("bstar must be finite")
    if value == 0.0:
        return " 00000-0"
    exponent = int(np.floor(np.log10(abs(value)))) + 1
    mantissa = int(round(abs(value) / (10.0 ** exponent) * 1e5))
    if mantissa >= 100000:
        mantissa //= 10
        exponent += 1
    if abs(exponent) > 9:
        raise ValueError("bstar magnitude cannot be represented in a TLE")
    return f"{'-' if value < 0 else ' '}{mantissa:05d}{'+' if exponent >= 0 else '-'}{abs(exponent)}"


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
    :type satellite: ADCS.satellite_hardware.satellite.satellite.Satellite
    :return: Estimated B* in the standard TLE convention.
    :rtype: float
    :raises ValueError: If drag geometry or a positive mass is unavailable.
    """
    if satellite is None:
        return None
    drag = next((d for d in satellite.disturbances if isinstance(d, Drag_Disturbance)), None)
    if drag is None:
        raise ValueError("satellite must include a Drag_Disturbance to derive B*")
    if satellite.mass <= 0:
        raise ValueError("satellite mass must be positive to derive B*")
    # Isotropic attitude average of projected area: integral(max(0,n·u)) dΩ / 4π = 1/4.
    average_cd_area = float(np.sum(drag.CDs * drag.areas)) / 4.0
    # SGP4's reference density is 2.461e-5 kg/m², yielding standard TLE B* units.
    return 0.5 * 2.461e-5 * average_cd_area / satellite.mass


class SGP4Orbit(Orbit):
    r"""Attitude-independent SGP4 orbit propagated from an initial state.

    An approximate TLE is made from ``os0.to_TLE()`` and used to initialize
    SGP4. Since a TLE contains model-dependent mean elements, this step does
    not preserve the initial osculating trajectory exactly and is not a TLE fit.
    SGP4 then advances the satellite independently of the attitude loop.

    If ``bstar`` is omitted and a satellite is provided, B* is estimated from
    its isotropic average drag area. For faces :math:`i` with area :math:`A_i`
    and drag coefficient :math:`C_{D,i}`,

    .. math::

       \overline{C_D A} = \frac{1}{4}\sum_i C_{D,i}A_i, \qquad
       B^* = \frac{1}{2}\rho_0\frac{\overline{C_D A}}{m}.

    Here :math:`m` is satellite mass and :math:`\rho_0=2.461\times10^{-5}`
    is the SGP4 reference-density convention. This is a rough attitude-average
    estimate; a manually supplied B* takes precedence. The SGP4/TLE B* field
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
        omitted. It must include a ``Drag_Disturbance``.
    :type satellite: ADCS.satellite_hardware.satellite.satellite.Satellite or None
    :param bstar: Manual SGP4/TLE B* value, in inverse Earth radii. If supplied,
        this value overrides the geometry-based estimate.
    :type bstar: float or None
    :raises TypeError: If ``os0`` is not one :class:`Orbital_State`.
    :raises ValueError: If B* is not finite, or geometry-derived B* is invalid.
    """

    def __init__(self, os0, end_time=None, dt=None, fast=True, verbose=True,
                 zonal_J=0, magnetic_model=None, environment=None,
                 satellite=None, bstar=None):
        """Initialize SGP4 from an initial osculating state.

        See the class documentation for parameter definitions and the B* model.

        :return: ``None``; initializes a queryable orbit.
        :rtype: None
        """
        if not isinstance(os0, Orbital_State):
            raise TypeError("SGP4Orbit requires one initial Orbital_State")
        if bstar is None:
            bstar = _average_bstar(satellite)
            if bstar is None:
                bstar = 0.0
        self.bstar = float(bstar)
        if not np.isfinite(self.bstar):
            raise ValueError("bstar must be finite")
        line1, line2 = os0.to_TLE()
        line1 = line1[:53] + _format_bstar(self.bstar) + line1[61:68]
        self._tle_lines = (_tle_checksum(line1), line2)
        super().__init__(os0, end_time=end_time, dt=dt, fast=fast, verbose=verbose,
                         zonal_J=zonal_J, magnetic_model=magnetic_model,
                         environment=environment)

    def _propagate_trajectory(self, os0, times_arr, mu_e, R_e, J2coeff, Jcoeffs, *, verbose=True):
        """Propagate one requested epoch at a time with the SGP4 TLE model."""
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
