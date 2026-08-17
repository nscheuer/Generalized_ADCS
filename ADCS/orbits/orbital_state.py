# orbital_state.py
__all__ = ["Orbital_State"]

import numpy as np
import warnings
import ppigrf
from skyfield import api, units, positionlib, toposlib, framelib
from datetime import timezone
from typing import Dict, Tuple, Optional

from ADCS.orbits.density_model import DensityModel
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.universal_constants import EarthConstants, TimeConstants
from ADCS.helpers.math_helpers import normalize, rot_mat, drotmatTvecdq, ddrotmatTvecdqdq
from ADCS.state import State

_I3 = np.eye(3)
_I6 = np.eye(6)

# Unit vector along the ECI/ECEF z-axis (Earth spin axis), used by the zonal
# gravity perturbation.
_ZHAT = np.array([0.0, 0.0, 1.0])


def _normalize_zonal_J(zonal_J, max_degree=None):
    r"""
    Validate and normalize the requested highest zonal harmonic degree.

    ``zonal_J=0`` disables all zonal harmonics (pure two-body dynamics).
    Positive values represent the highest degree included, so ``zonal_J=2``
    means J2 only and ``zonal_J=6`` means J2 through J6.
    """
    if max_degree is None:
        max_degree = len(EarthConstants.Jcoeffs) + 1

    value = int(zonal_J)
    if value == 0:
        return 0
    if 2 <= value <= int(max_degree):
        return value
    raise ValueError(f"zonal_J must be 0 or an integer from 2 through {int(max_degree)}, got {zonal_J!r}")


def _legendre_p_and_dp(s, max_degree):
    r"""
    Evaluate Legendre polynomials and their first derivatives at ``s``.

    Uses the standard upward recurrences

    .. math::
        n P_n = (2n-1)\, s\, P_{n-1} - (n-1) P_{n-2}, \qquad
        P_n' = n\, P_{n-1} + s\, P_{n-1}',

    which are stable everywhere on :math:`s\in[-1,1]` (no division by
    :math:`1-s^2`, so they hold over the poles) and are analytic in ``s``,
    so the routine is safe for complex-step differentiation.

    :param s:
        Argument of the polynomials (``sin`` of geocentric latitude). May be a
        real or complex scalar.
    :type s: float or complex

    :param max_degree:
        Highest polynomial degree to evaluate.
    :type max_degree: int

    :return:
        Tuple ``(P, dP)`` of lists indexed by degree ``0..max_degree``.
    :rtype: tuple[list, list]
    """
    P = [None] * (max_degree + 1)
    dP = [None] * (max_degree + 1)
    P[0] = 1.0 + 0.0 * s  # preserve dtype (float or complex) of s
    dP[0] = 0.0 * s
    if max_degree >= 1:
        P[1] = s
        dP[1] = 1.0 + 0.0 * s
    for n in range(2, max_degree + 1):
        P[n] = ((2 * n - 1) * s * P[n - 1] - (n - 1) * P[n - 2]) / n
        dP[n] = n * P[n - 1] + s * dP[n - 1]
    return P, dP


def _zonal_perturbation_accel(R, mu_e, R_e, coeffs, start_degree):
    r"""
    Perturbing acceleration from zonal gravity harmonics of degree ``>= start_degree``.

    The acceleration is the gradient of the zonal perturbing potential

    .. math::
        \Phi_n = -\frac{\mu}{r} J_n \left(\frac{R_e}{r}\right)^n P_n(z/r),

    summed over degrees ``start_degree .. start_degree + len(coeffs) - 1``. For
    ``start_degree == 2`` the degree-2 term equals the classical closed-form J2
    acceleration exactly; this routine is used for the higher (degree ``>= 3``)
    harmonics while J2 keeps its dedicated closed form.

    The implementation contains no non-analytic operations (no ``norm``/``abs``),
    so passing a complex ``R`` yields exact complex-step derivatives.

    :param R:
        Position vector in ECI frame [km]. Real or complex.
    :type R: numpy.ndarray

    :param mu_e:
        Earth gravitational parameter [km^3/s^2].
    :type mu_e: float

    :param R_e:
        Earth reference radius [km].
    :type R_e: float

    :param coeffs:
        Unnormalized zonal coefficients :math:`J_n` for the consecutive degrees
        starting at ``start_degree``.
    :type coeffs: sequence[float]

    :param start_degree:
        Degree of the first coefficient in ``coeffs``.
    :type start_degree: int

    :return:
        Perturbing acceleration [km/s^2] with the same dtype as ``R``.
    :rtype: numpy.ndarray
    """
    coeffs = np.asarray(coeffs)
    if coeffs.size == 0:
        return np.zeros(3, dtype=R.dtype if hasattr(R, "dtype") else float)

    R = np.asarray(R).reshape(3)
    z = R[2]
    r2 = R @ R
    r = np.sqrt(r2)
    s = z / r

    max_degree = start_degree + coeffs.size - 1
    P, dP = _legendre_p_and_dp(s, max_degree)

    accel = np.zeros(3, dtype=R.dtype)
    for k in range(coeffs.size):
        n = start_degree + k
        Jn = coeffs[k]
        if Jn == 0:
            continue
        # a_n = -mu * Jn * R_e^n * grad( r^-(n+1) * P_n(z/r) )
        #     = c_n * (alpha_n * R + beta_n * zhat)
        c_n = -mu_e * Jn * R_e**n
        alpha_n = -(n + 1) * r ** (-(n + 3)) * P[n] - z * r ** (-(n + 4)) * dP[n]
        beta_n = r ** (-(n + 2)) * dP[n]
        accel = accel + c_n * (alpha_n * R + beta_n * _ZHAT)
    return accel


def _zonal_perturbation_accel_jac(R, mu_e, R_e, coeffs, start_degree):
    r"""
    Jacobian :math:`\partial a/\partial R` of :func:`_zonal_perturbation_accel`.

    Computed by complex-step differentiation of the (analytic) acceleration,
    which is accurate to machine precision and free of subtractive cancellation.

    :param R:
        Position vector in ECI frame [km].
    :type R: numpy.ndarray

    :param mu_e:
        Earth gravitational parameter [km^3/s^2].
    :type mu_e: float

    :param R_e:
        Earth reference radius [km].
    :type R_e: float

    :param coeffs:
        Unnormalized zonal coefficients for consecutive degrees from ``start_degree``.
    :type coeffs: sequence[float]

    :param start_degree:
        Degree of the first coefficient in ``coeffs``.
    :type start_degree: int

    :return:
        3x3 Jacobian of the perturbing acceleration with respect to position.
    :rtype: numpy.ndarray
    """
    coeffs = np.asarray(coeffs)
    if coeffs.size == 0:
        return np.zeros((3, 3), dtype=float)

    R0 = np.asarray(R, dtype=complex).reshape(3)
    h = 1e-30
    jac = np.zeros((3, 3), dtype=float)
    for j in range(3):
        Rp = R0.copy()
        Rp[j] = Rp[j] + 1j * h
        accel = _zonal_perturbation_accel(Rp, mu_e, R_e, coeffs, start_degree)
        jac[:, j] = np.asarray(accel).imag / h
    return jac


class Orbital_State:
    r"""
    Complete dynamical and environmental representation of a spacecraft orbital state.

    The state is defined by inertial position and velocity in the ECI/ICRF frame at
    an epoch expressed in J2000 centuries. The class also stores commonly-used
    derived quantities and environment vectors, including Earth-fixed coordinates,
    Sun vector, geomagnetic field vector (IGRF), atmospheric density, and frame
    transforms.

    Notes
    -----
    The ``fast`` argument is accepted for backward compatibility but is ignored:
    this implementation always computes real Sun and geomagnetic field vectors and
    frame transforms.

    :param ephem:
        Ephemeris object used for Earth and Sun position queries.
        If ``None``, a new :class:`~ADCS.orbits.ephemeris.Ephemeris` is constructed.
    :type ephem: Ephemeris

    :param J2000:
        Epoch in Julian centuries since J2000.
    :type J2000: float

    :param R:
        Position vector in ECI frame [km].
    :type R: numpy.ndarray

    :param V:
        Velocity vector in ECI frame [km/s].
    :type V: numpy.ndarray

    :param S:
        Optional Sun position vector in ECI frame [km]. If ``None``, it is computed.
    :type S: numpy.ndarray or None

    :param B:
        Optional geomagnetic field vector in ECI frame [T]. If ``None``, it is computed.
    :type B: numpy.ndarray or None

    :param rho:
        Optional atmospheric density [kg/m^3]. If ``None``, it is computed using
        :class:`~ADCS.orbits.density_model.DensityModel`.
    :type rho: float or None

    :param density_model:
        Atmospheric density interpolation model. If ``None``, a new
        :class:`~ADCS.orbits.density_model.DensityModel` is constructed.
    :type density_model: DensityModel or None

    :param fast:
        Backward-compatible parameter (ignored). Real environment vectors are always
        computed.
    :type fast: bool

    """

    def __init__(
        self,
        ephem: Ephemeris,
        J2000: float,
        R: np.ndarray,
        V: np.ndarray,
        S: Optional[np.ndarray] = None,
        B: Optional[np.ndarray] = None,
        rho: Optional[float] = None,
        density_model: Optional[DensityModel] = None,
        fast: bool = False,
    ) -> None:
        r"""
        Initialize a fully defined orbital state.

        :param ephem:
            Ephemeris object used for Earth and Sun position queries.
        :type ephem: Ephemeris

        :param J2000:
            Epoch in Julian centuries since J2000.
        :type J2000: float

        :param R:
            Position vector in ECI frame [km].
        :type R: numpy.ndarray

        :param V:
            Velocity vector in ECI frame [km/s].
        :type V: numpy.ndarray

        :param S:
            Optional Sun vector in ECI frame [km]. If not provided, it is computed.
        :type S: numpy.ndarray or None

        :param B:
            Optional geomagnetic field vector in ECI frame [T]. If not provided, it is computed.
        :type B: numpy.ndarray or None

        :param rho:
            Optional atmospheric density [kg/m^3]. If not provided, it is computed.
        :type rho: float or None

        :param density_model:
            Atmospheric density model.
        :type density_model: DensityModel or None

        :param fast:
            Backward-compatible parameter (ignored).
        :type fast: bool

        :return:
            ``None``
        :rtype: None

        """
        _ = fast  # ignored (kept for backward compatibility)

        self.ephem = Ephemeris() if ephem is None else ephem
        self.ts = self.ephem.ts

        self.J2000 = float(J2000)
        self.R = np.asarray(R, dtype=float).reshape(3)
        self.V = np.asarray(V, dtype=float).reshape(3)

        self.mu_e = EarthConstants.mu_e
        self.R_e = EarthConstants.R_e
        self.J2coeff = EarthConstants.J2coeff
        # Unnormalized zonal harmonic coefficients [J2, J3, J4, J5, J6]; the
        # higher (degree >= 3) terms are only applied when propagation is asked
        # for zonal_J > 2.
        self.Jcoeffs = EarthConstants.Jcoeffs

        # NOTE: j2000_to_tai() returns J2000*36525 + 2451545.0. JD 2451545.0 is
        # the J2000.0 epoch in TT (2000-01-01 12:00:00 TT), so this Julian date
        # is on the TT scale and must be passed to ts.tt_jd, NOT ts.tai_jd
        # (which mislabeled it as TAI -> a constant TT-TAI = 32.184 s epoch
        # error in every time-dependent quantity: ECEF, Sun, IGRF). The legacy
        # attribute name `self.TAI` is kept for serialization/back-compat.
        self.TAI = self.j2000_to_tai()
        t_sf = self.ts.tt_jd(self.TAI)

        # Construct a Skyfield ICRF position for downstream functionality (e.g. is_sunlit).
        pos = units.Distance(km=self.R.tolist())
        vel_sf = units.Velocity(km_per_s=self.V.tolist())
        self.sf_pos = positionlib.ICRF(
            pos.au,
            velocity_au_per_d=vel_sf.au_per_d,
            t=t_sf,
            center=399,
            target=0,
        )

        # Cache a naive UTC datetime for ppigrf compatibility.
        dt_aware = self.sf_pos.t.astimezone(timezone.utc)
        self.datetime = dt_aware.replace(tzinfo=None)

        # ECI <-> ECEF rotation at this time (Skyfield)
        self._R_eci2ecef = framelib.itrs.rotation_at(self.sf_pos.t)
        self._R_ecef2eci = self._R_eci2ecef.T

        # ECEF position and geocentric coordinates
        self.ECEF = self._R_eci2ecef @ self.R

        r_ecef = float(np.linalg.norm(self.ECEF))
        th = float(np.arccos(self.ECEF[2] / r_ecef))
        ph = float(np.arctan2(self.ECEF[1], self.ECEF[0]))
        self.geocentric = np.array([r_ecef, th, ph], dtype=float)

        # Local geocentric basis in ECEF
        self._n_ecef = normalize(self.ECEF)
        self._svec = normalize(np.cross(np.array([0.0, 0.0, 1.0]), self._n_ecef))
        self._tvec = normalize(np.cross(self._svec, self._n_ecef))
        self._ecef_to_geo = np.vstack([self._n_ecef, self._tvec, self._svec])

        # Geographic position and ENU transform (always computed)
        self.sf_geo_pos = api.wgs84.geographic_position_of(self.sf_pos)
        self.LLA = np.array(
            [
                self.sf_geo_pos.latitude.radians,
                self.sf_geo_pos.longitude.radians,
                self.sf_geo_pos.elevation.km,
            ],
            dtype=float,
        )
        R_eci_to_ecef = self.sf_geo_pos.rotation_at(self.sf_pos.t)
        R_ecef_to_enu = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
        self.ECI2ENUmat = R_ecef_to_enu @ R_eci_to_ecef

        # Atmospheric density model
        self.density_model = density_model if density_model is not None else DensityModel()

        # Sun vector (ECI)
        if S is not None:
            self.S = np.asarray(S, dtype=float).reshape(3)
        else:
            self.S = self.get_sun_eci()

        # Magnetic field (ECI)
        if B is not None:
            self.B = np.asarray(B, dtype=float).reshape(3)
        else:
            self.B = self.get_b_eci()

        # Atmospheric density
        if rho is not None:
            self.rho = float(rho)
        else:
            altitude_from_core = float(np.linalg.norm(self.R))
            self.rho = float(self.density_model.interpolate(altitude_from_core - EarthConstants.R_e))

        self.vecs: Dict[str, np.ndarray] | None = None
        self._last_x: np.ndarray | None = None

    def copy(self):
        r"""
        Return a deep copy of the orbital state.

        This implementation avoids recomputing environment vectors and frame
        transforms.

        :return:
            Independent copy of the orbital state.
        :rtype: Orbital_State

        """
        out = Orbital_State.__new__(Orbital_State)

        out.ephem = self.ephem
        out.ts = self.ts

        out.J2000 = float(self.J2000)
        out.R = self.R.copy()
        out.V = self.V.copy()

        out.mu_e = self.mu_e
        out.R_e = self.R_e
        out.J2coeff = self.J2coeff
        out.Jcoeffs = np.array(self.Jcoeffs, dtype=float, copy=True)

        out.TAI = float(self.TAI)
        out.sf_pos = self.sf_pos  # treated as immutable for practical purposes
        out.datetime = self.datetime

        out._R_eci2ecef = np.array(self._R_eci2ecef, dtype=float, copy=True)
        out._R_ecef2eci = np.array(self._R_ecef2eci, dtype=float, copy=True)

        out.ECEF = self.ECEF.copy()
        out.geocentric = self.geocentric.copy()

        out._n_ecef = self._n_ecef.copy()
        out._svec = self._svec.copy()
        out._tvec = self._tvec.copy()
        out._ecef_to_geo = np.array(self._ecef_to_geo, dtype=float, copy=True)

        out.sf_geo_pos = getattr(self, "sf_geo_pos", None)
        out.LLA = np.array(self.LLA, dtype=float, copy=True)
        out.ECI2ENUmat = np.array(self.ECI2ENUmat, dtype=float, copy=True)

        out.density_model = self.density_model
        out.S = self.S.copy()
        out.B = self.B.copy()
        out.rho = float(self.rho)

        out.vecs = None
        out._last_x = None

        # Preserve any cached sunlit flag if present
        if hasattr(self, "_sunlit"):
            out._sunlit = bool(getattr(self, "_sunlit"))

        return out

    def average(self, orbital_state_2, ratio: float = 0.5, fast: bool = False):
        r"""
        Linearly interpolate between two orbital states.

        :param orbital_state_2:
            Second orbital state.
        :type orbital_state_2: Orbital_State

        :param ratio:
            Interpolation ratio in [0, 1].
        :type ratio: float

        :param fast:
            Backward-compatible parameter (ignored).
        :type fast: bool

        :return:
            Interpolated orbital state.
        :rtype: Orbital_State

        """
        _ = fast  # ignored
        os2 = orbital_state_2
        a = 1.0 - float(ratio)
        b = float(ratio)

        # Build the result via __new__ to skip the heavy Skyfield constructor.
        # All frame-rotation matrices and environment vectors are linearly
        # interpolated from the two endpoint states that were already fully
        # constructed during orbit propagation.
        out = Orbital_State.__new__(Orbital_State)

        out.ephem = self.ephem
        out.ts = self.ts

        out.J2000 = a * self.J2000 + b * os2.J2000
        out.R = a * self.R + b * os2.R
        out.V = a * self.V + b * os2.V
        out.S = a * self.S + b * os2.S
        out.B = a * self.B + b * os2.B
        out.rho = a * self.rho + b * os2.rho

        out.mu_e = self.mu_e
        out.R_e = self.R_e
        out.J2coeff = self.J2coeff
        out.Jcoeffs = np.array(self.Jcoeffs, dtype=float, copy=True)

        out.TAI = a * self.TAI + b * os2.TAI
        out.sf_pos = self.sf_pos          # approximate; only used by is_sunlit
        out.datetime = self.datetime       # approximate

        out._R_eci2ecef = a * self._R_eci2ecef + b * os2._R_eci2ecef
        out._R_ecef2eci = a * self._R_ecef2eci + b * os2._R_ecef2eci

        out.ECEF = out._R_eci2ecef @ out.R
        out.geocentric = a * self.geocentric + b * os2.geocentric

        out._n_ecef = normalize(out.ECEF)
        out._svec = a * self._svec + b * os2._svec
        out._tvec = a * self._tvec + b * os2._tvec
        out._ecef_to_geo = a * self._ecef_to_geo + b * os2._ecef_to_geo

        out.sf_geo_pos = getattr(self, "sf_geo_pos", None)
        out.LLA = a * self.LLA + b * os2.LLA
        out.ECI2ENUmat = a * self.ECI2ENUmat + b * os2.ECI2ENUmat

        out.density_model = self.density_model

        out.vecs = None
        out._last_x = None
        out._cached_sunlit = getattr(self, '_cached_sunlit', None)

        # Cache sunlit from nearest endpoint to avoid expensive Skyfield ephemeris call
        if hasattr(self, '_sunlit'):
            out._sunlit = self._sunlit
        elif hasattr(os2, '_sunlit'):
            out._sunlit = os2._sunlit

        # Propagate Jacobian skip flag
        out._skip_jacobians = getattr(self, '_skip_jacobians', False)

        return out

    @staticmethod
    def _orbit_dynamics_raw(
        R: np.ndarray,
        V: np.ndarray,
        mu_e: float,
        R_e: float,
        J2coeff: float,
        zonal_J: int = 2,
        Jcoeffs: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Compute raw orbital dynamics.

        :param R:
            Position vector in ECI frame.
        :type R: numpy.ndarray

        :param V:
            Velocity vector in ECI frame.
        :type V: numpy.ndarray

        :param mu_e:
            Earth gravitational parameter.
        :type mu_e: float

        :param R_e:
            Earth mean radius.
        :type R_e: float

        :param J2coeff:
            Earth J2 coefficient.
        :type J2coeff: float

        :param zonal_J:
            Highest zonal harmonic degree to include. ``0`` disables zonals,
            ``2`` includes only J2, and larger values include every zonal term
            up to that degree.
        :type zonal_J: int

        :param Jcoeffs:
            Unnormalized zonal coefficients ``[J2, J3, ...]``. When omitted,
            :data:`EarthConstants.Jcoeffs` is used.
        :type Jcoeffs: numpy.ndarray or None

        :return:
            Tuple of position and velocity derivatives.
        :rtype: tuple[numpy.ndarray, numpy.ndarray]

        """
        R = np.asarray(R, dtype=float).reshape(3)
        V = np.asarray(V, dtype=float).reshape(3)
        zonal_J = _normalize_zonal_J(zonal_J)
        Jcoeffs = EarthConstants.Jcoeffs if Jcoeffs is None else np.asarray(Jcoeffs, dtype=float)

        r2 = float(np.dot(R, R))
        rn = float(np.sqrt(r2))
        r3 = r2 * rn

        v_dot = -mu_e * R / r3

        if zonal_J >= 2:
            xk, yk, zk = R
            z2 = zk * zk
            factor = 1.5 * J2coeff * mu_e * R_e * R_e / (rn**5)
            common = 5.0 * z2 / r2
            a_J2 = factor * np.array([xk * (common - 1.0), yk * (common - 1.0), zk * (common - 3.0)], dtype=float)
            v_dot = v_dot + a_J2

            higher_zonals = Jcoeffs[1 : zonal_J - 1]
            if np.size(higher_zonals) > 0:
                v_dot = v_dot + _zonal_perturbation_accel(R, mu_e, R_e, higher_zonals, start_degree=3)

        r_dot = V
        return r_dot, v_dot

    @staticmethod
    def _orbit_dynamics_jacobians_raw(
        R: np.ndarray,
        mu_e: float,
        R_e: float,
        J2coeff: float,
        zonal_J: int = 2,
        Jcoeffs: Optional[np.ndarray] = None,
    ):
        r"""
        Compute Jacobians of orbital dynamics.

        :param R:
            Position vector in ECI frame.
        :type R: numpy.ndarray

        :param mu_e:
            Earth gravitational parameter.
        :type mu_e: float

        :param R_e:
            Earth mean radius.
        :type R_e: float

        :param J2coeff:
            Earth J2 coefficient.
        :type J2coeff: float

        :param zonal_J:
            Highest zonal harmonic degree to include. ``0`` disables zonals,
            ``2`` includes only J2, and larger values include every zonal term
            up to that degree.
        :type zonal_J: int

        :param Jcoeffs:
            Unnormalized zonal coefficients ``[J2, J3, ...]``. When omitted,
            :data:`EarthConstants.Jcoeffs` is used.
        :type Jcoeffs: numpy.ndarray or None

        :return:
            Partial derivatives of dynamics.
        :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]

        """
        R = np.asarray(R, dtype=float).reshape(3)
        zonal_J = _normalize_zonal_J(zonal_J)
        Jcoeffs = EarthConstants.Jcoeffs if Jcoeffs is None else np.asarray(Jcoeffs, dtype=float)

        rn = float(np.linalg.norm(R))
        nr = R / rn
        zk = float(R[2])

        drd_dr = np.zeros((3, 3), dtype=float)
        drd_dv = _I3.copy()
        dvd_dv = np.zeros((3, 3), dtype=float)

        dvd_dr = -mu_e * (_I3 - 3.0 * np.outer(nr, nr)) / rn**3

        if zonal_J >= 2:
            rn2 = rn * rn
            j2_mult = np.diagflat(np.array([1.0, 1.0, 3.0]) * rn2 - np.ones(3) * 5.0 * zk * zk)

            coeff = mu_e * (1.0 / rn**7.0) * (J2coeff * R_e**2) * (3.0 / 2.0)
            unit_z = np.array([0.0, 0.0, 1.0], dtype=float)

            dvd_dr += -coeff * (
                -7.0 * (np.outer(R, R @ j2_mult)) / rn2
                + j2_mult
                + 2.0 * (np.outer(-5.0 * zk * unit_z + R, R) + 2.0 * zk * np.outer(R, unit_z))
            )

            higher_zonals = Jcoeffs[1 : zonal_J - 1]
            if np.size(higher_zonals) > 0:
                dvd_dr += _zonal_perturbation_accel_jac(R, mu_e, R_e, higher_zonals, start_degree=3)

        return drd_dr, drd_dv, dvd_dr, dvd_dv

    def orbit_dynamics(self, zonal_J: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Compute translational orbital dynamics at the current state.

        :param zonal_J:
            Highest zonal harmonic degree to include. ``0`` disables zonals,
            ``2`` includes only J2, and larger values include every zonal term
            up to that degree.
        :type zonal_J: int

        :return:
            Time derivatives of position and velocity.
        :rtype: tuple[numpy.ndarray, numpy.ndarray]

        """
        return self._orbit_dynamics_raw(
            self.R, self.V, self.mu_e, self.R_e, self.J2coeff, zonal_J,
            Jcoeffs=self.Jcoeffs,
        )

    def orbit_dynamics_jacobians(self, zonal_J: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Compute Jacobians of the translational dynamics.

        :param zonal_J:
            Highest zonal harmonic degree to include. ``0`` disables zonals,
            ``2`` includes only J2, and larger values include every zonal term
            up to that degree.
        :type zonal_J: int

        :return:
            Jacobian matrices of the dynamics.
        :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]

        """
        return self._orbit_dynamics_jacobians_raw(
            self.R, self.mu_e, self.R_e, self.J2coeff, zonal_J,
            Jcoeffs=self.Jcoeffs,
        )

    def propagate_orbit(self, dt: float, fast: bool = True, zonal_J: int = 2):
        r"""
        Propagate the orbital state forward using first-order integration.

        :param dt:
            Time step in seconds.
        :type dt: float

        :param fast:
            Backward-compatible parameter (ignored).
        :type fast: bool

        :param zonal_J:
            Highest zonal harmonic degree to include. ``0`` disables zonals,
            ``2`` includes only J2, and larger values include every zonal term
            up to that degree.
        :type zonal_J: int

        :return:
            Propagated orbital state.
        :rtype: Orbital_State

        """
        _ = fast  # ignored
        r_ECI = self.R
        v_ECI = self.V

        k1a, k1b = self._orbit_dynamics_raw(
            r_ECI, v_ECI, self.mu_e, self.R_e, self.J2coeff, zonal_J, Jcoeffs=self.Jcoeffs
        )

        r_out = r_ECI + k1a * float(dt)
        v_out = v_ECI + k1b * float(dt)
        j2000 = self.J2000 + (float(dt) / TimeConstants.cent2sec)

        return Orbital_State(self.ephem, j2000, r_out, v_out, S=None, B=None, rho=None, density_model=self.density_model, fast=False)

    def propagate_orbit_rk4(self, dt: float, fast: bool = True, zonal_J: int = 2):
        r"""
        Propagate the orbital state using fourth-order Runge–Kutta integration.

        :param dt:
            Time step in seconds.
        :type dt: float

        :param fast:
            Backward-compatible parameter (ignored).
        :type fast: bool

        :param zonal_J:
            Highest zonal harmonic degree to include. ``0`` disables zonals,
            ``2`` includes only J2, and larger values include every zonal term
            up to that degree.
        :type zonal_J: int

        :return:
            Propagated orbital state.
        :rtype: Orbital_State

        """
        _ = fast  # ignored
        dt = float(dt)

        r0 = self.R
        v0 = self.V

        k1a, k1b = self._orbit_dynamics_raw(r0, v0, self.mu_e, self.R_e, self.J2coeff, zonal_J, Jcoeffs=self.Jcoeffs)
        k2a, k2b = self._orbit_dynamics_raw(r0 + 0.5 * dt * k1a, v0 + 0.5 * dt * k1b, self.mu_e, self.R_e, self.J2coeff, zonal_J, Jcoeffs=self.Jcoeffs)
        k3a, k3b = self._orbit_dynamics_raw(r0 + 0.5 * dt * k2a, v0 + 0.5 * dt * k2b, self.mu_e, self.R_e, self.J2coeff, zonal_J, Jcoeffs=self.Jcoeffs)
        k4a, k4b = self._orbit_dynamics_raw(r0 + dt * k3a, v0 + dt * k3b, self.mu_e, self.R_e, self.J2coeff, zonal_J, Jcoeffs=self.Jcoeffs)

        r_out = r0 + (dt / 6.0) * (k1a + 2.0 * k2a + 2.0 * k3a + k4a)
        v_out = v0 + (dt / 6.0) * (k1b + 2.0 * k2b + 2.0 * k3b + k4b)

        j2000 = self.J2000 + (dt / TimeConstants.cent2sec)

        return Orbital_State(self.ephem, j2000, r_out, v_out, S=None, B=None, rho=None, density_model=self.density_model, fast=False)

    def propagate_jacobians(self, dt: float, zonal_J: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Propagate state transition Jacobians using first-order integration.

        :param dt:
            Time step in seconds.
        :type dt: float

        :param zonal_J:
            Highest zonal harmonic degree to include. ``0`` disables zonals,
            ``2`` includes only J2, and larger values include every zonal term
            up to that degree.
        :type zonal_J: int

        :return:
            State transition Jacobian blocks.
        :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]

        """
        dt = float(dt)
        drd0__dr0, drd0__dv0, dvd0__dr0, dvd0__dv0 = self._orbit_dynamics_jacobians_raw(
            self.R, self.mu_e, self.R_e, self.J2coeff, zonal_J, Jcoeffs=self.Jcoeffs
        )

        dr1__dr0 = _I3 + dt * drd0__dr0
        dr1__dv0 = dt * drd0__dv0
        dv1__dr0 = dt * dvd0__dr0
        dv1__dv0 = _I3 + dt * dvd0__dv0

        return dr1__dr0, dr1__dv0, dv1__dr0, dv1__dv0

    def propagate_jacobians_rk4(self, dt: float, zonal_J: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Propagate state transition Jacobians using RK4 integration.

        :param dt:
            Time step in seconds.
        :type dt: float

        :param zonal_J:
            Highest zonal harmonic degree to include. ``0`` disables zonals,
            ``2`` includes only J2, and larger values include every zonal term
            up to that degree.
        :type zonal_J: int

        :return:
            State transition Jacobian blocks.
        :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]

        """
        dt = float(dt)

        r0 = self.R
        v0 = self.V

        rd0, vd0 = self._orbit_dynamics_raw(r0, v0, self.mu_e, self.R_e, self.J2coeff, zonal_J, Jcoeffs=self.Jcoeffs)
        drd0__dr0, drd0__dv0, dvd0__dr0, dvd0__dv0 = self._orbit_dynamics_jacobians_raw(
            r0, self.mu_e, self.R_e, self.J2coeff, zonal_J, Jcoeffs=self.Jcoeffs
        )
        dsd0__ds0 = np.block([[drd0__dr0, dvd0__dr0], [drd0__dv0, dvd0__dv0]])

        r1 = r0 + rd0 * 0.5 * dt
        v1 = v0 + vd0 * 0.5 * dt
        ds1__ds0 = _I6 + 0.5 * dt * dsd0__ds0

        rd1, vd1 = self._orbit_dynamics_raw(r1, v1, self.mu_e, self.R_e, self.J2coeff, zonal_J, Jcoeffs=self.Jcoeffs)
        drd1__dr1, drd1__dv1, dvd1__dr1, dvd1__dv1 = self._orbit_dynamics_jacobians_raw(
            r1, self.mu_e, self.R_e, self.J2coeff, zonal_J, Jcoeffs=self.Jcoeffs
        )
        dsd1__ds1 = np.block([[drd1__dr1, dvd1__dr1], [drd1__dv1, dvd1__dv1]])
        dsd1__ds0 = ds1__ds0 @ dsd1__ds1

        r2 = r0 + rd1 * 0.5 * dt
        v2 = v0 + vd1 * 0.5 * dt
        ds2__ds0 = _I6 + 0.5 * dt * dsd1__ds0

        rd2, vd2 = self._orbit_dynamics_raw(r2, v2, self.mu_e, self.R_e, self.J2coeff, zonal_J, Jcoeffs=self.Jcoeffs)
        drd2__dr2, drd2__dv2, dvd2__dr2, dvd2__dv2 = self._orbit_dynamics_jacobians_raw(
            r2, self.mu_e, self.R_e, self.J2coeff, zonal_J, Jcoeffs=self.Jcoeffs
        )
        dsd2__ds2 = np.block([[drd2__dr2, dvd2__dr2], [drd2__dv2, dvd2__dv2]])
        dsd2__ds0 = ds2__ds0 @ dsd2__ds2

        r3 = r0 + rd2 * dt
        v3 = v0 + vd2 * dt
        ds3__ds0 = _I6 + dt * dsd2__ds0

        rd3, vd3 = self._orbit_dynamics_raw(r3, v3, self.mu_e, self.R_e, self.J2coeff, zonal_J, Jcoeffs=self.Jcoeffs)
        drd3__dr3, drd3__dv3, dvd3__dr3, dvd3__dv3 = self._orbit_dynamics_jacobians_raw(
            r3, self.mu_e, self.R_e, self.J2coeff, zonal_J, Jcoeffs=self.Jcoeffs
        )
        dsd3__ds3 = np.block([[drd3__dr3, dvd3__dr3], [drd3__dv3, dvd3__dv3]])
        dsd3__ds0 = ds3__ds0 @ dsd3__ds3

        dsd4__ds0 = _I6 + (dt / 6.0) * (dsd0__ds0 + 2.0 * dsd1__ds0 + 2.0 * dsd2__ds0 + dsd3__ds0)

        return (
            dsd4__ds0[0:3, 0:3],
            dsd4__ds0[3:6, 0:3],
            dsd4__ds0[0:3, 3:6],
            dsd4__ds0[3:6, 3:6],
        )

    def eci_to_ecef(self, vec: np.ndarray) -> np.ndarray:
        r"""
        Transform a vector from ECI to ECEF coordinates.

        :param vec:
            Vector in ECI frame.
        :type vec: numpy.ndarray

        :return:
            Vector in ECEF frame.
        :rtype: numpy.ndarray

        """
        return self._R_eci2ecef @ np.asarray(vec, dtype=float)

    def ecef_to_eci(self, vec: np.ndarray) -> np.ndarray:
        r"""
        Transform a vector from ECEF to ECI coordinates.

        :param vec:
            Vector in ECEF frame.
        :type vec: numpy.ndarray

        :return:
            Vector in ECI frame.
        :rtype: numpy.ndarray

        """
        return self._R_ecef2eci @ np.asarray(vec, dtype=float)

    def ecef_to_geocentric(self, vec: np.ndarray) -> np.ndarray:
        r"""
        Transform an ECEF vector to local geocentric coordinates.

        :param vec:
            Vector in ECEF frame.
        :type vec: numpy.ndarray

        :return:
            Vector in geocentric basis.
        :rtype: numpy.ndarray

        """
        # _ecef_to_geo = [n; t; s] (basis vectors as rows). geocentric_to_ecef
        # is v0*n+v1*t+v2*s = _ecef_to_geo.T @ v, so the inverse (ECEF ->
        # geocentric) is _ecef_to_geo @ vec, NOT _ecef_to_geo.T @ vec (the
        # transpose was not the inverse of geocentric_to_ecef).
        return self._ecef_to_geo @ np.asarray(vec, dtype=float)

    def geocentric_to_ecef(self, vec: np.ndarray) -> np.ndarray:
        r"""
        Transform a geocentric vector to ECEF coordinates.

        :param vec:
            Vector in geocentric basis.
        :type vec: numpy.ndarray

        :return:
            Vector in ECEF frame.
        :rtype: numpy.ndarray

        """
        v = np.asarray(vec, dtype=float).reshape(3)
        return v[0] * self._n_ecef + v[1] * self._tvec + v[2] * self._svec

    def eci_to_enu(self, vec: np.ndarray) -> np.ndarray:
        r"""
        Transform a vector from ECI to local ENU frame.

        :param vec:
            Vector in ECI frame.
        :type vec: numpy.ndarray

        :return:
            Vector in ENU frame.
        :rtype: numpy.ndarray

        """
        return np.asarray(vec, dtype=float) @ self.ECI2ENUmat.T

    def enu_to_eci(self, vec: np.ndarray) -> np.ndarray:
        r"""
        Transform a vector from ENU to ECI frame.

        :param vec:
            Vector in ENU frame.
        :type vec: numpy.ndarray

        :return:
            Vector in ECI frame.
        :rtype: numpy.ndarray

        """
        return np.asarray(vec, dtype=float) @ self.ECI2ENUmat

    def get_b_eci(self) -> np.ndarray:
        r"""
        Compute the geomagnetic field vector in the ECI frame.

        :return:
            Magnetic field vector [Tesla].
        :rtype: numpy.ndarray

        """
        r = float(self.geocentric[0])
        theta_rad = float(self.geocentric[1])
        phi_rad = float(self.geocentric[2])

        b_r, b_th, b_ph = ppigrf.igrf_gc(
            r,
            theta_rad * 180.0 / np.pi,
            phi_rad * 180.0 / np.pi,
            self.datetime,
        )

        b_array = np.array(
            [
                np.asarray(b_r, dtype=float).reshape(-1)[0],
                np.asarray(b_th, dtype=float).reshape(-1)[0],
                np.asarray(b_ph, dtype=float).reshape(-1)[0],
            ],
            dtype=float,
        )
        b_ecef = self.geocentric_to_ecef(b_array)
        b_eci = self.ecef_to_eci(b_ecef)
        return b_eci * 1e-9

    def j2000_to_tai(self):
        r"""
        Convert J2000 centuries to TAI Julian date.

        :return:
            TAI Julian date.
        :rtype: float

        """
        return self.J2000 * 36525.0 + 2451545.0

    def get_sun_eci(self) -> np.ndarray:
        r"""
        Compute the Sun position vector in the ECI frame.

        :return:
            Sun vector in ECI coordinates [km].
        :rtype: numpy.ndarray

        """
        t = self.ts.tt_jd(self.TAI)  # TT scale (see __init__ note on self.TAI)
        sun_icrf = self.ephem.earth.at(t).observe(self.ephem.sun).apparent()
        return np.asarray(sun_icrf.position.km, dtype=float).reshape(3)

    def update_vecs(self, x: State) -> None:
        r"""
        Update body-frame vectors and their derivatives from a structured state.

        :param x:
            Spacecraft state containing the attitude quaternion.
        :type x: ADCS.state.State

        :return:
            ``None``
        :rtype: None

        """
        if not isinstance(x, State):
            raise TypeError(f"x must be a State, got {type(x).__name__}")
        q0 = x.q

        R = self.R
        V = self.V
        B = self.B
        S = self.S
        rho = self.rho

        rmat_ECI2B = rot_mat(q0).T
        R_B = rmat_ECI2B @ R
        B_B = rmat_ECI2B @ B
        S_B = rmat_ECI2B @ S
        V_B = rmat_ECI2B @ V

        # Jacobians/Hessians are only needed when an estimator is in the loop.
        # Skip them when _skip_jacobians is set to avoid ~25% overhead.
        if getattr(self, '_skip_jacobians', False):
            _z3x4 = np.zeros((3, 4))
            _z3x4x4 = np.zeros((3, 4, 4))
            dR_B__dq = _z3x4
            dB_B__dq = _z3x4
            dV_B__dq = _z3x4
            dS_B__dq = _z3x4
            ddR_B__dqdq = _z3x4x4
            ddB_B__dqdq = _z3x4x4
            ddV_B__dqdq = _z3x4x4
            ddS_B__dqdq = _z3x4x4
        else:
            dR_B__dq = drotmatTvecdq(q0, R)
            dB_B__dq = drotmatTvecdq(q0, B)
            dV_B__dq = drotmatTvecdq(q0, V)
            dS_B__dq = drotmatTvecdq(q0, S)
            ddR_B__dqdq = ddrotmatTvecdqdq(q0, R)
            ddB_B__dqdq = ddrotmatTvecdqdq(q0, B)
            ddV_B__dqdq = ddrotmatTvecdqdq(q0, V)
            ddS_B__dqdq = ddrotmatTvecdqdq(q0, S)

        self.vecs = {
            "b": B_B,
            "r": R_B,
            "s": S_B,
            "v": V_B,
            "rho": rho,
            "db": dB_B__dq,
            "ds": dS_B__dq,
            "dv": dV_B__dq,
            "dr": dR_B__dq,
            "ddb": ddB_B__dqdq,
            "dds": ddS_B__dqdq,
            "ddv": ddV_B__dqdq,
            "ddr": ddR_B__dqdq,
        }
        # Body-frame environment vectors depend only on attitude. Keeping the
        # quaternion as the cache key avoids allocating a full state vector on
        # every sensor and disturbance lookup.
        self._last_x = x.q.copy()

    def get_state_vector(self, x: Optional[State]) -> Dict[str, np.ndarray]:
        r"""
        Retrieve cached or updated body-frame vectors.

        :param x:
            Current spacecraft state, or ``None`` to return initialized cached vectors.
        :type x: ADCS.state.State or None

        :return:
            Dictionary of vectors and derivatives.
        :rtype: dict[str, numpy.ndarray]

        """
        if x is None:
            if self._last_x is None:
                raise ValueError("x is required before body-frame vectors have been initialized")
            return self.vecs
        if not isinstance(x, State):
            raise TypeError(f"x must be a State, got {type(x).__name__}")
        if self._last_x is None or not np.array_equal(x.q, self._last_x):
            self.update_vecs(x=x)
        return self.vecs

    def is_sunlit(self) -> bool:
        r"""
        Determine whether the spacecraft is illuminated by the Sun.

        :return:
            ``True`` if sunlit, ``False`` otherwise.
        :rtype: bool

        """
        if hasattr(self, "_sunlit"):
            return bool(getattr(self, "_sunlit"))
        result = bool(self.sf_pos.is_sunlit(self.ephem.planets))
        self._sunlit = result
        return result

    def to_dict(self) -> dict:
        r"""
        Serialize the orbital state to a dictionary.

        :return:
            Dictionary representation of the orbital state.
        :rtype: dict

        """
        return {"J2000": self.J2000, "R": self.R, "V": self.V, "S": self.S, "B": self.B, "rho": self.rho}

    @classmethod
    def from_dict(cls, d: dict, ephem: Ephemeris, density_model: DensityModel | None = None, fast: bool = True):
        r"""
        Construct an orbital state from a dictionary.

        Notes
        -----
        The ``fast`` argument is accepted for backward compatibility but is ignored.

        :param d:
            Dictionary containing orbital state fields.
        :type d: dict

        :param ephem:
            Ephemeris object.
        :type ephem: Ephemeris

        :param density_model:
            Atmospheric density model.
        :type density_model: DensityModel or None

        :param fast:
            Backward-compatible parameter (ignored).
        :type fast: bool

        :return:
            Reconstructed orbital state.
        :rtype: Orbital_State

        """
        _ = fast  # ignored
        return cls(
            ephem=ephem,
            J2000=d["J2000"],
            R=d["R"],
            V=d["V"],
            S=d.get("S"),
            B=d.get("B"),
            rho=d.get("rho"),
            density_model=density_model,
            fast=False,
        )
