__all__ = ["Coordinate_Goal"]

import numpy as np
from typing import Tuple

from ADCS.CONOPS.goals import Vector_Goal
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.helpers.math_helpers import normalize

class Coordinate_Goal(Vector_Goal):
    """
    Ground-pointing goal defined by fixed geodetic coordinates.

    The :class:`Coordinate_Goal` represents a pointing objective toward a
    fixed location on the rotating Earth, specified by geodetic latitude,
    longitude, and altitude (WGS84).

    Internally, the target location is converted once to Earth-Centered
    Earth-Fixed (ECEF) coordinates and stored. At runtime, the ECEF
    position is transformed into the inertial (ECI) frame using the
    current orbital state.

    This goal produces:
    
    * a unit inertial line-of-sight vector from the spacecraft to the target
    * a reference angular velocity required to keep the target centered,
      accounting for both spacecraft motion and Earth rotation

    Parameters
    ----------
    lat : float
        Geodetic latitude in degrees.
    lon : float
        Geodetic longitude in degrees.
    alt : float
        Geodetic altitude above the WGS84 ellipsoid in kilometers.

    Attributes
    ----------
    lat_deg : float
        Target latitude in degrees.
    lon_deg : float
        Target longitude in degrees.
    alt_km : float
        Target altitude in kilometers.
    target_ecef : numpy.ndarray
        Target position expressed in ECEF coordinates [km].

    See Also
    --------
    :class:`~ADCS.goals.goal.Goal`
    :class:`~ADCS.orbits.orbital_state.Orbital_State`
    """
    def __init__(self, lat: float, lon: float, alt: float) -> None:
        """
        Initialize a coordinate-based ground target goal.

        The provided geodetic coordinates are immediately converted to
        Earth-Centered Earth-Fixed (ECEF) coordinates and stored for
        efficient reuse during runtime.

        Parameters
        ----------
        lat : float
            Geodetic latitude in degrees.
        lon : float
            Geodetic longitude in degrees.
        alt : float
            Geodetic altitude in kilometers.
        """
        self.lat_deg = lat
        self.lon_deg = lon
        self.alt_km = alt

        self.target_ecef = self._geodetic_to_ecef(lat, lon, alt)

    def _geodetic_to_ecef(self, lat_deg: float, lon_deg: float, alt_km: float) -> np.ndarray:
        """
        Convert geodetic coordinates to ECEF coordinates (WGS84).

        This method converts latitude, longitude, and altitude referenced
        to the WGS84 ellipsoid into Earth-Centered Earth-Fixed (ECEF)
        Cartesian coordinates.

        Parameters
        ----------
        lat_deg : float
            Geodetic latitude in degrees.
        lon_deg : float
            Geodetic longitude in degrees.
        alt_km : float
            Altitude above the WGS84 ellipsoid in kilometers.

        Returns
        -------
        numpy.ndarray
            ECEF position vector ``[x, y, z]`` in kilometers.

        Notes
        -----
        The WGS84 parameters used are:

        .. math::

            a &= 6378.137 \, \text{km} \\
            f &= \frac{1}{298.257223563} \\
            e^2 &= 2f - f^2

        This method is intended for internal use only.
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

        This method computes the inertial line-of-sight vector from the
        spacecraft to the specified ground target, as well as the reference
        angular velocity required to maintain continuous pointing.

        The target is assumed to be fixed in the Earth-Centered Earth-Fixed
        (ECEF) frame and rotating with the Earth. The spacecraft state is
        provided via the current orbital state.

        Parameters
        ----------
        os0 : Orbital_State
            Current orbital state, provided as an
            :class:`~ADCS.orbits.orbital_state.Orbital_State`, containing
            spacecraft position, velocity, and time information.

        Returns
        -------
        r_goal_eci : numpy.ndarray, shape (3,)
            Normalized unit vector in the ECI frame pointing from the
            spacecraft to the ground target.
        w_ref_eci : numpy.ndarray, shape (3,)
            Reference angular velocity vector in the ECI frame required to
            keep the target centered in the body frame.

        Notes
        -----
        The reference angular velocity is computed as:

        .. math::

            \boldsymbol{\omega}_{ref}
            =
            \frac{\mathbf{r}_{rel} \times \mathbf{v}_{rel}}
                {\lVert \mathbf{r}_{rel} \rVert^2}

        where:

        * :math:`\mathbf{r}_{rel}` is the relative position vector
        from spacecraft to target
        * :math:`\mathbf{v}_{rel}` is the relative velocity accounting
        for Earth rotation and spacecraft motion

        Earth rotation is modeled using a constant angular velocity:

        .. math::

            \boldsymbol{\omega}_{Earth}
            =
            [0, 0, 7.2921159 \times 10^{-5}] \, \text{rad/s}

        See Also
        --------
        :meth:`ADCS.goals.goal.Goal.to_ref`
        :meth:`ADCS.orbits.orbital_state.Orbital_State.ecef_to_eci`
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

        return r_goal_eci, w_ref_eci