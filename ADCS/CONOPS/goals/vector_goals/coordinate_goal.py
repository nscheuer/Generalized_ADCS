__all__ = ["Coordinate_Goal"]

import numpy as np
from typing import Tuple

from ADCS.CONOPS.goals import Vector_Goal
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.helpers.math_helpers import normalize

class Coordinate_Goal(Vector_Goal):
    r"""
    Ground-pointing goal defined by fixed geodetic coordinates.

    This class implements a vector-alignment goal that points a
    spacecraft body-frame vector toward a fixed location on the Earth
    defined by geodetic latitude, longitude, and altitude using the
    WGS84 reference ellipsoid.

    The target location is fixed in the Earth-Centered Earth-Fixed
    (ECEF) frame and rotates with the Earth. At runtime, the target
    position is transformed into the inertial frame using the current
    orbital state.

    The goal mapping implemented by this class is

    .. math::

        G_{\mathrm{coord}}(\mathcal{O}(t)) =
        \left(
            \hat{\mathbf{r}}_{\mathrm{LOS}}(t),
            \boldsymbol{\omega}_{\mathrm{ref}}(t)
        \right)

    where :math:`\hat{\mathbf{r}}_{\mathrm{LOS}}` is the inertial
    line-of-sight unit vector from the spacecraft to the ground target
    and :math:`\boldsymbol{\omega}_{\mathrm{ref}}` is the inertial
    angular velocity required to maintain continuous pointing.

    See Also
    --------
    :class:`~ADCS.goals.goal.Goal`
    :class:`~ADCS.CONOPS.goals.Vector_Goal`
    :class:`~ADCS.orbits.orbital_state.Orbital_State`

    """
    def __init__(self, lat: float, lon: float, alt: float, boresight_name: str | None = None) -> None:
        r"""
        Initialize a coordinate-based ground target goal.

        The provided geodetic coordinates are converted to
        Earth-Centered Earth-Fixed coordinates and stored internally for
        reuse during runtime.

        :param lat:
            Geodetic latitude in degrees.
        :type lat:
            float

        :param lon:
            Geodetic longitude in degrees.
        :type lon:
            float

        :param alt:
            Geodetic altitude above the WGS84 ellipsoid in kilometers.
        :type alt:
            float

        :param boresight_name:
            Optional name of the boresight to use from the satellite's
            boresight dictionary. If ``None``, the first available boresight
            is selected.
        :type boresight_name:
            str | None

        :return:
            None
        :rtype:
            None

        """
        super().__init__(boresight_name=boresight_name)
        self.lat_deg = lat
        self.lon_deg = lon
        self.alt_km = alt

        self.target_ecef = self._geodetic_to_ecef(lat, lon, alt)

    def _geodetic_to_ecef(self, lat_deg: float, lon_deg: float, alt_km: float) -> np.ndarray:
        r"""
        Convert geodetic coordinates to Earth-Centered Earth-Fixed coordinates.

        The conversion is performed using the WGS84 reference ellipsoid.
        Let :math:`(\phi, \lambda, h)` denote geodetic latitude,
        longitude, and altitude, respectively.

        The prime vertical radius of curvature is

        .. math::

            N(\phi) = \frac{a}{\sqrt{1 - e^2 \sin^2 \phi}}

        The resulting ECEF position vector is

        .. math::

            \mathbf{r}_{\mathrm{ECEF}} =
            \begin{bmatrix}
            (N + h)\cos\phi\cos\lambda \\
            (N + h)\cos\phi\sin\lambda \\
            (N(1 - e^2) + h)\sin\phi
            \end{bmatrix}

        where the WGS84 constants are

        .. math::

            a = 6378.137\ \mathrm{km}, \quad
            f = \frac{1}{298.257223563}, \quad
            e^2 = 2f - f^2

        :param lat_deg:
            Geodetic latitude in degrees.
        :type lat_deg:
            float

        :param lon_deg:
            Geodetic longitude in degrees.
        :type lon_deg:
            float

        :param alt_km:
            Altitude above the WGS84 ellipsoid in kilometers.
        :type alt_km:
            float

        :return:
            ECEF position vector in kilometers.
        :rtype:
            numpy.ndarray

        """
        lat_rad = np.radians(lat_deg)
        lon_rad = np.radians(lon_deg)

        # WGS84 Constants
        a = 6378.137            # Semi-major axis [km]
        f = 1.0 / 298.257223563 # Flattening
        e2 = 2*f - f**2         # Square of eccentricity

        N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)

        x = (N + alt_km) * np.cos(lat_rad) * np.cos(lon_rad)
        y = (N + alt_km) * np.cos(lat_rad) * np.sin(lon_rad)
        z = (N * (1 - e2) + alt_km) * np.sin(lat_rad)

        return np.array([x, y, z])
    
    def to_ref(self, os0: Orbital_State) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Compute inertial reference vectors for ground target tracking.

        The target position is assumed fixed in the ECEF frame and is
        transformed into the inertial frame using the current orbital
        state. The inertial line-of-sight vector is computed as the
        relative position between spacecraft and target.

        The reference angular velocity is computed from the relative
        motion of the line-of-sight vector as

        .. math::

            \boldsymbol{\omega}_{\mathrm{ref}} =
            \frac{\mathbf{r}_{\mathrm{rel}} \times \mathbf{v}_{\mathrm{rel}}}
                 {\lVert \mathbf{r}_{\mathrm{rel}} \rVert^2}

        where:

        - :math:`\mathbf{r}_{\mathrm{rel}}` is the relative position
          from spacecraft to target

        - :math:`\mathbf{v}_{\mathrm{rel}}` is the relative velocity
          accounting for both spacecraft motion and Earth rotation

        Earth rotation is modeled using the constant angular velocity

        .. math::

            \boldsymbol{\omega}_{\oplus} =
            \begin{bmatrix}
            0 \\
            0 \\
            7.2921159 \times 10^{-5}
            \end{bmatrix}
            \ \mathrm{rad/s}

        :param os0:
            Current orbital state.
        :type os0:
            Orbital_State

        :return:
            Inertial line-of-sight unit vector and reference angular
            velocity.
        :rtype:
            Tuple[numpy.ndarray, numpy.ndarray]

        """
        # 1. Get Satellite Position and Velocity in ECI
        r_sat_eci = os0.R
        v_sat_eci = os0.V

        # 2. Get Target Position in ECI
        # Convert the stored fixed ECEF coordinate to ECI at current time
        r_target_eci = os0.ecef_to_eci(self.target_ecef)

        # 3. Calculate Target Velocity in ECI (due to Earth Rotation)
        # Earth rotation rate vector in rad/s (approx value, frame Z-axis)
        w_earth = np.array([0.0, 0.0, 7.2921159e-5]) 
        v_target_eci = np.cross(w_earth, r_target_eci)

        # 4. Calculate Relative Position (Line of Sight Vector)
        r_rel = r_target_eci - r_sat_eci
        r_dist_sq = np.dot(r_rel, r_rel)
        
        # Normalized pointing vector (Output 1)
        r_goal_eci = normalize(r_rel)

        # 5. Calculate Reference Angular Velocity (Output 2)
        # The angular velocity of the Line of Sight vector relative to inertial space
        # w_ref = (r_rel x v_rel) / |r_rel|^2
        v_rel = v_target_eci - v_sat_eci
        w_ref_eci = np.cross(r_rel, v_rel) / r_dist_sq

        r_ref = np.empty((4,))
        r_ref[0] = np.nan
        r_ref[1:] = r_goal_eci

        return r_ref, w_ref_eci