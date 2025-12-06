__all__ = ["Coordinate_Goal"]

import numpy as np
from typing import Tuple

from .goal import Goal
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.helpers.math_helpers import normalize

class Coordinate_Goal(Goal):
    def __init__(self, lat: float, lon: float, alt: float) -> None:
        self.lat_deg = lat
        self.lon_deg = lon
        self.alt_km = alt

        self.target_ecef = self._geodetic_to_ecef(lat, lon, alt)

    def _geodetic_to_ecef(self, lat_deg: float, lon_deg: float, alt_km: float) -> np.ndarray:
        """
        Converts Geodetic coordinates (WGS84) to ECEF coordinates.
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
    
    def to_ref(self, x_hat: np.ndarray, os0: Orbital_State) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Computes the inertial pointing vector and reference angular velocity.

        Calculates the vector from the satellite to the ground target in the ECI frame.
        It also calculates the required angular rate :math:`\omega_{ref}` to keep
        the target centered, accounting for the relative velocity between the 
        satellite and the rotating Earth surface.

        Parameters
        ----------
        x : np.ndarray
            Current estimated satellite state (size 7+).
            x[0:3] = Angular Velocity (not used for target generation)
            x[3:7] = Quaternion (not used for target generation)
        os0 : Orbital_State
            Current orbital state containing position, velocity, and time data.

        Returns
        -------
        r_goal_eci : np.ndarray (3,)
            Normalized unit vector in ECI frame pointing from Satellite -> Target.
        w_ref_eci : np.ndarray (3,)
            Reference angular velocity vector in ECI frame required to track the target.
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