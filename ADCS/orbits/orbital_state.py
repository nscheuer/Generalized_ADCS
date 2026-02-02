__all__ = ["Orbital_State"]

import numpy as np
import warnings
import ppigrf
from skyfield import api, units, positionlib, toposlib, framelib, vectorlib
from datetime import timezone
from typing import Dict, Tuple, Optional

from ADCS.orbits.density_model import DensityModel
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.universal_constants import EarthConstants, TimeConstants
from ADCS.helpers.math_helpers import normalize, rot_mat, drotmatTvecdq, ddrotmatTvecdqdq

_I3 = np.eye(3)
_I6 = np.eye(6)

class Orbital_State:
    r"""
    Complete dynamical and environmental representation of a spacecraft orbit state.

    This class encapsulates the translational orbital state of a spacecraft
    together with time, reference-frame transformations, environmental models,
    and auxiliary vectors required for guidance, navigation, and control (GNC)
    and attitude determination and control systems (ADCS).

    An :class:`~ADCS.orbits.orbital_state.Orbital_State` is defined primarily by
    its inertial position and velocity vectors expressed in the Earth-Centered
    Inertial (ECI / ICRF) frame at a given epoch. From this core state, the class
    derives:

    * Earth-fixed (ECEF) coordinates
    * Geocentric spherical coordinates
    * Sun direction vector
    * Geomagnetic field vector (IGRF)
    * Atmospheric density
    * Linearized orbital dynamics and Jacobians
    * Frame transformations (ECI, ECEF, ENU, body)

    Mathematical Model
    ------------------
    The orbital motion is governed by Newton’s equation with optional J2
    perturbation:

    .. math::

        \ddot{\mathbf{r}} =
        -\frac{\mu}{r^3}\mathbf{r}
        + \mathbf{a}_{J_2},

    where

    .. math::

        \mathbf{a}_{J_2}
        = \frac{3 J_2 \mu R_E^2}{2 r^5}
        \begin{bmatrix}
            x \left(5 \frac{z^2}{r^2} - 1\right) \\
            y \left(5 \frac{z^2}{r^2} - 1\right) \\
            z \left(5 \frac{z^2}{r^2} - 3\right)
        \end{bmatrix}.

    Environmental quantities such as atmospheric density
    :math:`\rho(h)` and geomagnetic field :math:`\mathbf{B}` are evaluated
    using empirical models via
    :class:`~ADCS.orbits.density_model.DensityModel` and IGRF.

    :param ephem:
        Planetary ephemeris object providing Earth and Sun states.
    :type ephem: Ephemeris

    :param J2000:
        Epoch expressed in Julian centuries since J2000.
    :type J2000: float

    :param R:
        Spacecraft position vector in ECI frame [km].
    :type R: numpy.ndarray

    :param V:
        Spacecraft velocity vector in ECI frame [km/s].
    :type V: numpy.ndarray

    :param S:
        Optional Sun direction vector in ECI frame [km].
    :type S: numpy.ndarray or None

    :param B:
        Optional geomagnetic field vector in ECI frame [T].
    :type B: numpy.ndarray or None

    :param rho:
        Optional atmospheric density [kg/m³].
    :type rho: float or None

    :param density_model:
        Atmospheric density interpolation model.
    :type density_model: DensityModel or None

    :param fast:
        If ``True``, skips expensive environment and frame computations.
    :type fast: bool

    """
    def __init__(self, ephem: Ephemeris, J2000: float, R: np.ndarray, V: np.ndarray, S: np.ndarray = None, B: np.ndarray = None, rho: float = None, density_model: DensityModel = None, fast: bool = False) -> None:
        r"""
        Initialize a fully defined orbital state.

        This constructor initializes the inertial orbital state and derives
        all dependent quantities such as Earth-fixed coordinates, Sun vector,
        geomagnetic field, and atmospheric density.

        :param ephem:
            Ephemeris object used for Sun and Earth position queries.
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
            Optional Sun vector in ECI frame.
        :type S: numpy.ndarray or None

        :param B:
            Optional geomagnetic field vector in ECI frame.
        :type B: numpy.ndarray or None

        :param rho:
            Optional atmospheric density [kg/m³].
        :type rho: float or None

        :param density_model:
            Atmospheric density model.
        :type density_model: DensityModel or None

        :param fast:
            Skip non-essential calculations if ``True``.
        :type fast: bool

        :return:
            ``None``
        :rtype: None

        """
        if ephem is None:
            self.ephem = Ephemeris()
        else:
            self.ephem = ephem
        self.ts = self.ephem.ts

        self.J2000 = J2000
        self.R = np.asarray(R, dtype=float)
        self.V = np.asarray(V, dtype=float)

        self.mu_e = EarthConstants.mu_e
        self.R_e = EarthConstants.R_e
        self.J2coeff = EarthConstants.J2coeff

        self.TAI = self.j2000_to_tai()
        pos_time = self.ts.tai_jd(self.TAI)

        pos: units.Distance = units.Distance(km=self.R.tolist())
        vel_sf: units.Velocity = units.Velocity(km_per_s=self.V.tolist())

        self.sf_pos: positionlib.ICRF = positionlib.ICRF(
            pos.au.tolist(),
            velocity_au_per_d=vel_sf.au_per_d.tolist(),
            t=pos_time,
            center=399,
            target=0
        )

        self.datetime = self.sf_pos.t.astimezone(timezone.utc).replace(tzinfo = None)

        # Precompute ECI <-> ECEF rotation
        self._R_eci2ecef = framelib.itrs.rotation_at(self.sf_pos.t)
        self._R_ecef2eci = self._R_eci2ecef.T

        self.ECEF = self._R_eci2ecef @ self.R

        r = np.linalg.norm(self.ECEF)
        th = np.arccos(self.ECEF[2]/r)
        ph = np.arctan2(self.ECEF[1], self.ECEF[0])
        self.geocentric = np.array([r, th, ph])

        # Precompute local geocentric basis
        self._n_ecef = normalize(self.ECEF)
        self._svec = normalize(np.cross(np.array([0.0, 0.0, 1.0]), self._n_ecef))
        self._tvec = normalize(np.cross(self._svec, self._n_ecef))
        self._ecef_to_geo = np.vstack([self._n_ecef, self._tvec, self._svec])

        # Geographic position and ENU transform
        if not fast:
            self.sf_geo_pos: toposlib.GeographicPosition = api.wgs84.geographic_position_of(self.sf_pos)
            self.LLA = np.array([self.sf_geo_pos.latitude.radians, self.sf_geo_pos.longitude.radians, self.sf_geo_pos.elevation.km])

            R_eci_to_ecef = self.sf_geo_pos.rotation_at(self.sf_pos.t)
            R_ecef_to_enu = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
            self.ECI2ENUmat: np.ndarray = R_ecef_to_enu @ R_eci_to_ecef

        # Atmospheric density model
        if density_model is not None:
            self.density_model = density_model
        else:
            self.density_model = DensityModel()

        # Sun Vector
        if S is not None:
            self.S = np.asarray(S, dtype=float)
        elif fast:
            self.S = np.zeros(3, dtype=float)
        else:
            self.S = self.get_sun_eci()

        # Magnetic field
        if B is not None:
            self.B = np.asarray(B, dtype=float)
        elif fast:
            self.B = np.zeros(3, dtype=float)
        else:
            self.B = self.get_b_eci()

        # Atmospheric density
        if rho is not None:
            self.rho = float(rho)
        elif fast or self.density_model is None:
            self.rho = 0.0
        else:
            altitude_from_core = np.linalg.norm(self.R)
            self.rho = self.density_model.interpolate(altitude_from_core - EarthConstants.R_e)

        self.vecs: Dict[str, np.ndarray] | None = None
        self._last_x: np.ndarray | None = None


    def copy(self):
        r"""
        Return a deep copy of the orbital state.

        :return:
            Independent copy of the orbital state.
        :rtype: Orbital_State

        """
        return self.average(self, 0)
    

    def average(self, orbital_state_2, ratio: float = 0.5, fast: bool = False):
        r"""
        Linearly interpolate between two orbital states.

        The interpolation is performed element-wise on all state quantities:

        .. math::

            \mathbf{x}_{\text{avg}} =
            (1-\alpha)\mathbf{x}_1 + \alpha \mathbf{x}_2.

        :param orbital_state_2:
            Second orbital state.
        :type orbital_state_2: Orbital_State

        :param ratio:
            Interpolation ratio :math:`\alpha \in [0,1]`.
        :type ratio: float

        :param fast:
            Skip non-essential calculations in the output state.
        :type fast: bool

        :return:
            Interpolated orbital state.
        :rtype: Orbital_State

        """

        os2 = orbital_state_2
        a = 1.0 - ratio
        b = ratio

        j2000 = a * self.J2000 + b * os2.J2000
        R = a * self.R + b * os2.R
        V = a * self.V + b * os2.V
        S = a * self.S + b * os2.S
        B = a * self.B + b * os2.B
        rho = a * self.rho + b * os2.rho

        if not np.all(self.density_model.altitude_range == os2.density_model.altitude_range):
            warnings.warn(
                "non-matching altitude range in atmospheric model between 2 orbital states"
            )
        if not np.all(self.density_model.rho_range == os2.density_model.rho_range):
            warnings.warn(
                "non-matching air density vs altitude in atmospheric model between 2 orbital states"
            )

        density_model = self.density_model
        return Orbital_State(ephem=self.ephem, J2000=j2000, R=R, V=V, S=S, B=B, rho=rho, density_model=density_model, fast=fast)
    

    @staticmethod
    def _orbit_dynamics_raw(R: np.ndarray, V: np.ndarray, mu_e: float, R_e: float, J2coeff: float, J2_perturbation_on: bool = True) -> Tuple[np.ndarray, np.ndarray]:
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

        :param J2_perturbation_on:
            Enable J2 perturbation.
        :type J2_perturbation_on: bool

        :return:
            Tuple of position and velocity derivatives.
        :rtype:
            tuple[numpy.ndarray, numpy.ndarray]

        """
        r2 = float(np.dot(R, R))
        rn = np.sqrt(r2)
        r3 = r2 * rn

        v_dot = -mu_e * R / r3

        if J2_perturbation_on:
            xk, yk, zk = R
            z2 = zk * zk
            factor = 1.5 * J2coeff * mu_e * R_e * R_e / (rn**5)
            common = 5.0 * z2 / r2
            a_J2 = factor * np.array(
                [
                    xk * (common - 1.0),
                    yk * (common - 1.0),
                    zk * (common - 3.0),
                ]
            )
            v_dot = v_dot + a_J2

        r_dot = V
        return r_dot, v_dot
    

    @staticmethod
    def _orbit_dynamics_jacobians_raw(R: np.ndarray, mu_e: float, R_e: float, J2coeff: float, J2_perturbation_on: bool = True):
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

        :param J2_perturbation_on:
            Enable J2 perturbation.
        :type J2_perturbation_on: bool

        :return:
            Partial derivatives of dynamics.
        :rtype:
            tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]

        """
        rn = np.linalg.norm(R)
        nr = R / rn
        zk = R[2]

        drd_dr = np.zeros((3, 3))
        drd_dv = _I3.copy()
        dvd_dv = np.zeros((3, 3))

        dvd_dr = -mu_e * (_I3 - 3.0 * np.outer(nr, nr)) / rn**3

        if J2_perturbation_on:
            rn2 = rn * rn
            j2_mult = np.diagflat(
                np.array([1.0, 1.0, 3.0]) * rn2 - np.ones(3) * 5.0 * zk * zk
            )

            coeff = mu_e * (1.0 / rn**7.0) * (J2coeff * R_e**2) * (3.0 / 2.0)
            unit_z = np.array([0.0, 0.0, 1.0])

            dvd_dr += -coeff * (
                -7.0 * (np.outer(R, R @ j2_mult)) / rn2
                + j2_mult
                + 2.0
                * (
                    np.outer(-5.0 * zk * unit_z + R, R)
                    + 2.0 * zk * np.outer(R, unit_z)
                )
            )

        return drd_dr, drd_dv, dvd_dr, dvd_dv
    

    def orbit_dynamics(self, J2_perturbation_on: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Compute translational orbital dynamics at the current state.

        :param J2_perturbation_on:
            Enable J2 perturbation.
        :type J2_perturbation_on: bool

        :return:
            Time derivatives of position and velocity.
        :rtype:
            tuple[numpy.ndarray, numpy.ndarray]

        """
        return self._orbit_dynamics_raw(R=self.R, V=self.V, mu_e=self.mu_e, R_e=self.R_e, J2coeff=self.J2coeff, J2_perturbation_on=J2_perturbation_on)
    

    def orbit_dynamics_jacobians(self, J2_perturbation_on: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Compute Jacobians of the translational dynamics.

        :param J2_perturbation_on:
            Enable J2 perturbation.
        :type J2_perturbation_on: bool

        :return:
            Jacobian matrices of the dynamics.
        :rtype:
            tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]

        """
        return self._orbit_dynamics_jacobians_raw(R=self.R, mu_e=self.mu_e, R_e=self.R_e, J2coeff=self.J2coeff, J2_perturbation_on=J2_perturbation_on)
    

    def propagate_orbit(self, dt: float, J2_perturbation_on: bool = True, fast: bool = True):
        r"""
        Propagate the orbital state forward using first-order integration.

        :param dt:
            Time step in seconds.
        :type dt: float

        :param J2_perturbation_on:
            Enable J2 perturbation.
        :type J2_perturbation_on: bool

        :param fast:
            Skip environment updates if ``True``.
        :type fast: bool

        :return:
            Propagated orbital state.
        :rtype: Orbital_State

        """
        r_ECI = self.R
        v_ECI = self.V

        k1a, k1b = self._orbit_dynamics_raw(r_ECI, v_ECI, self.mu_e, self.R_e, self.J2coeff, J2_perturbation_on)

        r_out = r_ECI + k1a * dt
        v_out = v_ECI + k1b * dt

        j2000 = self.J2000 + (dt/TimeConstants.cent2sec)

        return Orbital_State(self.ephem, j2000, r_out, v_out, S=None, B=None, rho=None, density_model=self.density_model, fast=fast)
    

    def propagate_orbit_rk4(self, dt: float, J2_perturbation_on: bool = True, fast: bool = True):
        r"""
        Propagate the orbital state using fourth-order Runge–Kutta integration.

        :param dt:
            Time step in seconds.
        :type dt: float

        :param J2_perturbation_on:
            Enable J2 perturbation.
        :type J2_perturbation_on: bool

        :param fast:
            Skip environment updates if ``True``.
        :type fast: bool

        :return:
            Propagated orbital state.
        :rtype: Orbital_State

        """
        r0 = self.R
        v0 = self.V

        k1a, k1b = self._orbit_dynamics_raw(
            r0,
            v0,
            self.mu_e,
            self.R_e,
            self.J2coeff,
            J2_perturbation_on,
        )

        k2a, k2b = self._orbit_dynamics_raw(
            r0 + 0.5 * dt * k1a,
            v0 + 0.5 * dt * k1b,
            self.mu_e,
            self.R_e,
            self.J2coeff,
            J2_perturbation_on,
        )

        k3a, k3b = self._orbit_dynamics_raw(
            r0 + 0.5 * dt * k2a,
            v0 + 0.5 * dt * k2b,
            self.mu_e,
            self.R_e,
            self.J2coeff,
            J2_perturbation_on,
        )

        k4a, k4b = self._orbit_dynamics_raw(
            r0 + dt * k3a,
            v0 + dt * k3b,
            self.mu_e,
            self.R_e,
            self.J2coeff,
            J2_perturbation_on,
        )

        r_out = r0 + (dt / 6.0) * (k1a + 2.0 * k2a + 2.0 * k3a + k4a)
        v_out = v0 + (dt / 6.0) * (k1b + 2.0 * k2b + 2.0 * k3b + k4b)

        j2000 = self.J2000 + (dt/TimeConstants.cent2sec)

        return Orbital_State(self.ephem, j2000, r_out, v_out, S=None, B=None, rho=None, density_model=self.density_model, fast=fast)
    
    
    def propagate_jacobians(self, dt: float, J2_perturbation_on: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Propagate state transition Jacobians using first-order integration.

        :param dt:
            Time step in seconds.
        :type dt: float

        :param J2_perturbation_on:
            Enable J2 perturbation.
        :type J2_perturbation_on: bool

        :return:
            State transition Jacobian blocks.
        :rtype:
            tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]

        """
        drd0__dr0, drd0__dv0, dvd0__dr0, dvd0__dv0 = self._orbit_dynamics_jacobians_raw(
            self.R,
            self.mu_e,
            self.R_e,
            self.J2coeff,
            J2_perturbation_on,
        )

        dr1__dr0 = _I3 + dt * drd0__dr0
        dr1__dv0 = dt * drd0__dv0
        dv1__dr0 = dt * dvd0__dr0
        dv1__dv0 = _I3 + dt * dvd0__dv0

        return dr1__dr0, dr1__dv0, dv1__dr0, dv1__dv0
    

    def propagate_jacobians_rk4(self, dt: float, J2_perturbation_on: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Propagate state transition Jacobians using RK4 integration.

        :param dt:
            Time step in seconds.
        :type dt: float

        :param J2_perturbation_on:
            Enable J2 perturbation.
        :type J2_perturbation_on: bool

        :return:
            State transition Jacobian blocks.
        :rtype:
            tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]

        """

        r0 = self.R
        v0 = self.V

        rd0, vd0 = self._orbit_dynamics_raw(
            r0,
            v0,
            self.mu_e,
            self.R_e,
            self.J2coeff,
            J2_perturbation_on,
        )
        drd0__dr0, drd0__dv0, dvd0__dr0, dvd0__dv0 = self._orbit_dynamics_jacobians_raw(
            r0,
            self.mu_e,
            self.R_e,
            self.J2coeff,
            J2_perturbation_on,
        )
        dsd0__ds0 = np.block([[drd0__dr0, dvd0__dr0], [drd0__dv0, dvd0__dv0]])

        r1 = r0 + rd0 * 0.5 * dt
        v1 = v0 + vd0 * 0.5 * dt
        ds1__ds0 = _I6 + 0.5 * dt * dsd0__ds0

        rd1, vd1 = self._orbit_dynamics_raw(
            r1,
            v1,
            self.mu_e,
            self.R_e,
            self.J2coeff,
            J2_perturbation_on,
        )
        drd1__dr1, drd1__dv1, dvd1__dr1, dvd1__dv1 = self._orbit_dynamics_jacobians_raw(
            r1,
            self.mu_e,
            self.R_e,
            self.J2coeff,
            J2_perturbation_on,
        )
        dsd1__ds1 = np.block([[drd1__dr1, dvd1__dr1], [drd1__dv1, dvd1__dv1]])
        dsd1__ds0 = ds1__ds0 @ dsd1__ds1

        r2 = r0 + rd1 * 0.5 * dt
        v2 = v0 + vd1 * 0.5 * dt
        ds2__ds0 = _I6 + 0.5 * dt * dsd1__ds0

        rd2, vd2 = self._orbit_dynamics_raw(
            r2,
            v2,
            self.mu_e,
            self.R_e,
            self.J2coeff,
            J2_perturbation_on,
        )
        drd2__dr2, drd2__dv2, dvd2__dr2, dvd2__dv2 = self._orbit_dynamics_jacobians_raw(
            r2,
            self.mu_e,
            self.R_e,
            self.J2coeff,
            J2_perturbation_on,
        )
        dsd2__ds2 = np.block([[drd2__dr2, dvd2__dr2], [drd2__dv2, dvd2__dv2]])
        dsd2__ds0 = ds2__ds0 @ dsd2__ds2

        r3 = r0 + rd2 * dt
        v3 = v0 + vd2 * dt
        ds3__ds0 = _I6 + dt * dsd2__ds0

        rd3, vd3 = self._orbit_dynamics_raw(
            r3,
            v3,
            self.mu_e,
            self.R_e,
            self.J2coeff,
            J2_perturbation_on,
        )
        drd3__dr3, drd3__dv3, dvd3__dr3, dvd3__dv3 = self._orbit_dynamics_jacobians_raw(
            r3,
            self.mu_e,
            self.R_e,
            self.J2coeff,
            J2_perturbation_on,
        )
        dsd3__ds3 = np.block([[drd3__dr3, dvd3__dr3], [drd3__dv3, dvd3__dv3]])
        dsd3__ds0 = ds3__ds0 @ dsd3__ds3

        r4 = r0 + (dt / 6.0) * (rd0 + 2.0 * rd1 + 2.0 * rd2 + rd3)
        v4 = v0 + (dt / 6.0) * (vd0 + 2.0 * vd1 + 2.0 * vd2 + vd3)
        _ = (r4, v4)  # keep variables (not used, but consistent with original intent)

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

        return self._R_eci2ecef @ vec

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
        return self._R_ecef2eci @ vec

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
        return self._ecef_to_geo.T @ vec

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
        return vec[0] * self._n_ecef + vec[1] * self._tvec + vec[2] * self._svec

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
        return vec @ self.ECI2ENUmat.T

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
        return vec @ self.ECI2ENUmat

    def get_b_eci(self) -> np.ndarray:
        r"""
        Compute the geomagnetic field vector in the ECI frame.

        :return:
            Magnetic field vector [Tesla].
        :rtype: numpy.ndarray

        """
        r = self.geocentric[0]
        theta_rad = self.geocentric[1]
        phi_rad = self.geocentric[2]

        # IGRF expects radius in km, theta and phi in degrees
        # geocentric[0] is already in km since R is passed in km
        b_r, b_th, b_ph = ppigrf.igrf_gc(
            r_km,
            theta_rad * 180.0 / np.pi,
            phi_rad * 180.0 / np.pi,
            self.datetime,
        )
        b_array = np.array([b_r, b_th, b_ph])

        b_ecef = self.geocentric_to_ecef(np.squeeze(b_array))
        b_eci = self.ecef_to_eci(b_ecef)
        return b_eci * 1e-9 # Returned in Tesla

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
        timescale_object: api.Timescale = self.ephem.ts
        pos_time = timescale_object.tai_jd(self.TAI)

        sun_icrf: positionlib.ICRF = self.ephem.earth.at(pos_time).observe(
            self.ephem.sun
        ).apparent()
        sun_eci: np.ndarray = sun_icrf.position.km
        return sun_eci

    def update_vecs(self, x: np.ndarray) -> None:
        r"""
        Update body-frame vectors and their derivatives from a state vector.

        :param x:
            Full spacecraft state vector including attitude quaternion.
        :type x: numpy.ndarray

        :return:
            ``None``
        :rtype: None

        """
        q0 = x[3:7]

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
        self._last_x = x

    def get_state_vector(self, x: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
        r"""
        Retrieve cached or updated body-frame vectors.

        :param x:
            Current spacecraft state vector.
        :type x: numpy.ndarray or None

        :return:
            Dictionary of vectors and derivatives.
        :rtype:
            dict[str, numpy.ndarray]

        """
        if not np.array_equal(x, self._last_x):
            self.update_vecs(x=x)
        return self.vecs

    def is_sunlit(self) -> bool:
        r"""
        Determine whether the spacecraft is illuminated by the Sun.

        :return:
            ``True`` if sunlit, ``False`` otherwise.
        :rtype: bool

        """
        return self.sf_pos.is_sunlit(self.ephem.planets)
    
    def to_dict(self) -> dict:
        r"""
        Serialize the orbital state to a dictionary.

        :return:
            Dictionary representation of the orbital state.
        :rtype: dict

        """
        return {
            "J2000": self.J2000,
            "R": self.R,
            "V": self.V,
            "S": self.S,
            "B": self.B,
            "rho": self.rho,
        }


    @classmethod
    def from_dict(cls, d: dict, ephem: Ephemeris, density_model: DensityModel | None = None, fast: bool = True):
        r"""
        Construct an orbital state from a dictionary.

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
            Skip non-essential calculations if ``True``.
        :type fast: bool

        :return:
            Reconstructed orbital state.
        :rtype: Orbital_State

        """
        return cls(
            ephem=ephem,
            J2000=d["J2000"],
            R=d["R"],
            V=d["V"],
            S=d.get("S"),
            B=d.get("B"),
            rho=d.get("rho"),
            density_model=density_model,
            fast=fast,
    )