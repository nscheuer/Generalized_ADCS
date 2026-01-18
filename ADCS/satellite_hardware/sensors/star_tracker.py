"""
Star tracker sensor model.

This module implements a star tracker sensor that measures the direction
of bright navigation stars in the spacecraft body frame. The sensor model
includes realistic noise characteristics and visibility constraints.

Mathematical Model:
    The star tracker measures the direction of a star in the body frame:

        b = A(q)^T @ s_ECI + noise

    where:
        b = measured star direction (body frame, unit vector)
        A(q) = attitude DCM (Direction Cosine Matrix) from quaternion
        s_ECI = true star direction in J2000 ECI frame

    The measurement Jacobian ∂b/∂q is computed using the existing
    drotmatTvecdq() function from math_helpers.

Noise Model:
    Star trackers have anisotropic noise: cross-boresight accuracy is
    typically better than roll accuracy. Typical values (1σ):
    - Cross-boresight (pitch/yaw): 1-30 arcsec
    - Roll (about boresight): 5-100 arcsec

    Reference: Liebe (2002), Section III

Visibility Constraints:
    Stars may not be visible due to:
    1. Field of view limits
    2. Earth occlusion (star behind Earth)
    3. Moon occlusion (star behind Moon)
    4. Sun exclusion (stray light blinds tracker)

    Reference: Vallado (2013), Section 5.3

References:
    [1] Markley, F.L. & Crassidis, J.L., "Fundamentals of Spacecraft
        Attitude Determination and Control", Springer (2014), Ch. 5
    [2] Shuster, M.D., "A Survey of Attitude Representations",
        Journal of the Astronautical Sciences, 41(4):439-517 (1993)
    [3] Liebe, C.C., "Star Trackers for Attitude Determination",
        IEEE AES Magazine, Vol. 10, No. 6 (1995)
    [4] Vallado, D.A., "Fundamentals of Astrodynamics and Applications",
        4th Ed., Microcosm Press (2013)
"""
from __future__ import annotations

__all__ = ["StarTracker"]

import numpy as np
from typing import Optional
from numpy.typing import NDArray

from ADCS.satellite_hardware.sensors.sensor import Sensor
from ADCS.satellite_hardware.sensors.star_catalog import StarCatalog, NavigationStar
from ADCS.helpers.math_helpers import drotmatTvecdq, quat2rotmat
from ADCS.orbits.orbital_state import Orbital_State


class StarTracker(Sensor):
    """Star tracker sensor model.

    Measures the direction of bright navigation stars in the body frame.
    Returns NaN when no star is visible (similar to SunSensor behavior).

    Attributes:
        boresight: Boresight direction in body frame (unit vector)
        fov: Field of view in radians (full angle)
        cross_noise_std: Cross-boresight noise standard deviation in radians
        roll_noise_std: Roll noise standard deviation in radians
        sun_exclusion: Sun exclusion angle in radians
        catalog: Star catalog for visible star lookup
        current_star: Currently tracked star (updated each measurement)

    Mathematical Model:
        Measurement: b = A(q)^T @ s_ECI
        Jacobian: ∂b/∂q = drotmatTvecdq(q, s_ECI), shape (4, 3)

    Example:
        >>> tracker = StarTracker(
        ...     boresight=np.array([0, 0, 1]),
        ...     fov=np.deg2rad(15),
        ...     cross_noise_std=10 * np.pi / (180 * 3600),  # 10 arcsec
        ...     roll_noise_std=50 * np.pi / (180 * 3600),   # 50 arcsec
        ... )
        >>> measurement = tracker.clean_reading(x, orbital_state)
    """

    output_length: int = 3  # 3D unit vector

    def __init__(
        self,
        boresight: NDArray[np.float64],
        fov: float,
        cross_noise_std: float,
        roll_noise_std: float,
        sun_exclusion: float = np.deg2rad(25.0),
        catalog: Optional[StarCatalog] = None
    ) -> None:
        """Initialize star tracker.

        Args:
            boresight: Boresight direction in body frame. Will be normalized.
            fov: Field of view in radians (full angle, not half-angle).
            cross_noise_std: Cross-boresight (pitch/yaw) noise standard
                deviation in radians.
            roll_noise_std: Roll (about boresight) noise standard deviation
                in radians.
            sun_exclusion: Sun exclusion angle in radians. Tracker is blinded
                when sun is closer than this angle to boresight.
                Default: 25 degrees. Typical range: 25-45 degrees.
            catalog: Star catalog to use. If None, creates default catalog.
        """
        self.boresight = np.asarray(boresight, dtype=np.float64)
        self.boresight = self.boresight / np.linalg.norm(self.boresight)

        self.fov = float(fov)
        self.cross_noise_std = float(cross_noise_std)
        self.roll_noise_std = float(roll_noise_std)
        self.sun_exclusion = float(sun_exclusion)

        self.catalog = catalog if catalog is not None else StarCatalog()
        self.current_star: Optional[NavigationStar] = None

        # Build rotation from body-z to boresight for anisotropic noise model
        self._R_noise = self._build_noise_rotation()

    def _build_noise_rotation(self) -> NDArray[np.float64]:
        """Build rotation matrix to align z-axis with boresight.

        This rotation is used to apply anisotropic noise correctly:
        - z-axis (boresight) gets roll noise
        - x/y axes (perpendicular to boresight) get cross-boresight noise

        Returns:
            3x3 rotation matrix that transforms from boresight-aligned
            frame to body frame.
        """
        z = np.array([0.0, 0.0, 1.0])

        if np.allclose(self.boresight, z):
            return np.eye(3)
        if np.allclose(self.boresight, -z):
            return np.diag([1.0, -1.0, -1.0])

        # Rodrigues' rotation formula
        v = np.cross(z, self.boresight)
        s = np.linalg.norm(v)
        c = np.dot(z, self.boresight)
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
        return R

    def _get_sun_eci(self, os: Orbital_State) -> Optional[NDArray[np.float64]]:
        """Get sun position in ECI from orbital state.

        Args:
            os: Orbital state object

        Returns:
            Sun position in ECI (km), or None if not available.
        """
        if hasattr(os, 'S') and os.S is not None:
            s = np.asarray(os.S, dtype=np.float64)
            if not np.allclose(s, 0):
                return s
        return None

    def _get_moon_eci(self, os: Orbital_State) -> Optional[NDArray[np.float64]]:
        """Get moon position in ECI from orbital state.

        Uses Skyfield via the ephemeris to compute Moon position.

        Args:
            os: Orbital state object

        Returns:
            Moon position in ECI (km), or None if ephemeris not available.
        """
        try:
            if hasattr(os, 'ephem') and os.ephem is not None:
                moon = os.ephem.planets['moon']
                moon_icrf = os.ephem.earth.at(os.sf_pos.t).observe(moon).apparent()
                return np.asarray(moon_icrf.position.km, dtype=np.float64)
        except (KeyError, AttributeError):
            pass
        return None

    def _select_star(
        self,
        q: NDArray[np.float64],
        os: Orbital_State
    ) -> Optional[NavigationStar]:
        """Select the brightest visible star.

        Accounts for all visibility constraints:
        1. Field of view limits
        2. Earth occlusion (star behind Earth)
        3. Moon occlusion (star behind Moon)
        4. Sun exclusion (tracker blinded by stray light)

        Args:
            q: Attitude quaternion (scalar-first convention)
            os: Orbital state for satellite position and celestial body positions

        Returns:
            Brightest visible NavigationStar, or None if none visible.
        """
        # Get attitude DCM and boresight in ECI
        A = quat2rotmat(q)
        boresight_eci = A @ self.boresight

        # Get satellite position (required for occlusion checks)
        r_sat_eci = os.R

        # Get sun and moon positions for occlusion/exclusion
        sun_eci = self._get_sun_eci(os)
        moon_eci = self._get_moon_eci(os)

        # Get visible stars with full occlusion checking
        visible = self.catalog.get_visible_stars(
            boresight_eci=boresight_eci,
            fov_rad=self.fov,
            r_sat_eci=r_sat_eci,
            sun_eci=sun_eci,
            moon_eci=moon_eci,
            sun_exclusion_rad=self.sun_exclusion
        )

        if not visible:
            return None

        # Select brightest (lowest visual magnitude)
        return min(visible, key=lambda s: s.vmag)

    def clean_reading(
        self,
        x: NDArray[np.float64],
        os: Orbital_State
    ) -> NDArray[np.float64]:
        """Compute clean (noiseless) star tracker measurement.

        Args:
            x: State vector with angular velocity at x[0:3] and
                quaternion at x[3:7]
            os: Orbital state providing satellite position and
                celestial body positions

        Returns:
            Star direction in body frame, shape (3,).
            Returns array of NaN if no star is visible.

        Mathematical Model:
            b = A(q)^T @ s_ECI

            where A(q) is the DCM from quaternion and s_ECI is the
            star direction in J2000 ECI frame.

        Reference:
            Markley & Crassidis (2014), Eq. 5.108
        """
        q = x[3:7]

        # Select a visible star
        star = self._select_star(q, os)
        if star is None:
            self.current_star = None
            return np.full(3, np.nan)

        self.current_star = star

        # Compute measurement: b = A(q)^T @ s_ECI
        A = quat2rotmat(q)
        b = A.T @ star.s_eci

        return b

    def noisy_reading(
        self,
        x: NDArray[np.float64],
        os: Orbital_State
    ) -> NDArray[np.float64]:
        """Compute noisy star tracker measurement.

        Applies anisotropic noise model with different cross-boresight
        and roll noise characteristics:
        - Cross-boresight (pitch/yaw): uses cross_noise_std
        - Roll (about boresight): uses roll_noise_std

        Args:
            x: State vector with quaternion at x[3:7]
            os: Orbital state

        Returns:
            Noisy star direction in body frame, shape (3,).
            Returns array of NaN if no star is visible.

        Reference:
            Liebe (1995), Section III: Noise characteristics
        """
        clean = self.clean_reading(x, os)
        if np.any(np.isnan(clean)):
            return clean

        # Generate anisotropic noise in boresight-aligned frame
        # x/y axes = cross-boresight (pitch/yaw)
        # z axis = boresight (roll)
        noise_aligned = np.array([
            np.random.normal(0, self.cross_noise_std),
            np.random.normal(0, self.cross_noise_std),
            np.random.normal(0, self.roll_noise_std)
        ])

        # Rotate noise to body frame
        noise_body = self._R_noise @ noise_aligned

        # Apply noise and renormalize to unit vector
        noisy = clean + noise_body
        noisy = noisy / np.linalg.norm(noisy)

        return noisy

    def basestate_jac(
        self,
        x: NDArray[np.float64],
        os: Orbital_State
    ) -> NDArray[np.float64]:
        """Compute Jacobian of measurement with respect to base state.

        The base state is [omega_x, omega_y, omega_z, q0, q1, q2, q3].
        The measurement only depends on the quaternion, not angular velocity.

        Args:
            x: State vector with quaternion at x[3:7]
            os: Orbital state (used to determine current star)

        Returns:
            Jacobian matrix of shape (7, 3):
            - Rows 0-2 (omega): zeros (measurement independent of omega)
            - Rows 3-6 (quaternion): ∂b/∂q

        Mathematical Derivation:
            b = A(q)^T @ s_ECI
            ∂b/∂q = drotmatTvecdq(q, s_ECI)^T

            The drotmatTvecdq function computes ∂(R^T @ v)/∂q with shape (4, 3).

        Reference:
            Shuster (1993), Eq. 168 for quaternion-DCM derivatives
        """
        if self.current_star is None:
            # No star visible - return zeros
            return np.zeros((7, self.output_length))

        q = x[3:7]
        s_eci = self.current_star.s_eci

        # drotmatTvecdq returns shape (4, 3): ∂(R^T @ v)/∂q_i for each component
        db_dq = drotmatTvecdq(q, s_eci)  # Shape (4, 3)

        # Build full Jacobian (7, 3)
        J = np.zeros((7, self.output_length))
        J[3:7, :] = db_dq  # Shape (4, 3)

        return J

    def bias_jac(
        self,
        x: NDArray[np.float64],
        os: Orbital_State
    ) -> NDArray[np.float64]:
        """Compute Jacobian with respect to bias states.

        Star tracker has no bias states in this model.

        Args:
            x: State vector (unused)
            os: Orbital state (unused)

        Returns:
            Empty array of shape (0, 3)
        """
        return np.zeros((0, self.output_length))

    @property
    def noise_covariance(self) -> NDArray[np.float64]:
        """Get measurement noise covariance matrix.

        Returns the anisotropic noise covariance in the body frame,
        accounting for the boresight direction.

        Returns:
            Noise covariance matrix of shape (3, 3)
        """
        # Covariance in boresight-aligned frame
        # x/y = cross-boresight, z = roll (boresight axis)
        R_aligned = np.diag([
            self.cross_noise_std**2,
            self.cross_noise_std**2,
            self.roll_noise_std**2
        ])

        # Rotate to body frame: R_body = R_noise @ R_aligned @ R_noise^T
        return self._R_noise @ R_aligned @ self._R_noise.T
