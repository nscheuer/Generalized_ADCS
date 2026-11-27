"""Geodetic coordinate pointing goal."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numba import njit

from GenADCS.backends.python.helpers.math.frames.conversions import (
    ecef_to_gcrf_matrix,
)
from GenADCS.backends.python.helpers.math.rotations.quaternion.conversions import (
    dcm_to_quaternion,
)

from .goal import Goal, validate_reference_state


@njit(cache=True)
def _coordinate_dcm(position, velocity, epoch_jd, target_ecef):
    target = ecef_to_gcrf_matrix(epoch_jd, epoch_jd) @ target_ecef
    relative = target - position
    distance = np.linalg.norm(relative)
    if distance == 0.0:
        raise ValueError("spacecraft position coincides with coordinate target")
    z_axis = relative / distance

    x_raw = velocity - np.dot(velocity, z_axis) * z_axis
    x_norm = np.linalg.norm(x_raw)
    if x_norm < 1e-12:
        position_norm = np.linalg.norm(position)
        if position_norm == 0.0:
            raise ValueError("coordinate goal roll reference is degenerate")
        radial = position / position_norm
        x_raw = radial - np.dot(radial, z_axis) * z_axis
        x_norm = np.linalg.norm(x_raw)
    if x_norm < 1e-12:
        raise ValueError("coordinate goal roll reference is degenerate")
    x_axis = x_raw / x_norm
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)

    inertial_to_body = np.empty((3, 3))
    inertial_to_body[0, :] = x_axis
    inertial_to_body[1, :] = y_axis
    inertial_to_body[2, :] = z_axis
    return inertial_to_body


@njit(cache=True)
def _coordinate_fmc_dcm(position, velocity, epoch_jd, target_ecef, fmc_level):
    """Point at the FMC tilt angle, using the coordinate for tilt azimuth."""
    target = ecef_to_gcrf_matrix(epoch_jd, epoch_jd) @ target_ecef
    radius = np.linalg.norm(position)
    if radius == 0.0:
        raise ValueError("coordinate FMC goal is undefined at zero position")
    radial = position / radius
    nadir = -radial

    # The target fixes the horizontal tilt azimuth at the spacecraft. If it
    # lies directly below, use projected ground velocity to resolve azimuth.
    target_direction = target - position
    azimuth = target_direction - np.dot(target_direction, radial) * radial
    azimuth_norm = np.linalg.norm(azimuth)
    earth_rate = np.array([0.0, 0.0, 7.2921159e-5])
    ground_velocity = velocity - np.cross(earth_rate, position)
    if azimuth_norm < 1e-12:
        azimuth = ground_velocity - np.dot(ground_velocity, radial) * radial
        azimuth_norm = np.linalg.norm(azimuth)
    if azimuth_norm < 1e-12:
        raise ValueError("coordinate FMC tilt azimuth is degenerate")
    azimuth /= azimuth_norm

    # FMC n sets the nadir tilt to acos(1/n), giving v*cos(theta)=v/n
    # when the tilt azimuth follows the ground-track direction.
    tilt = np.arccos(1.0 / fmc_level)
    z_axis = np.cos(tilt) * nadir + np.sin(tilt) * azimuth

    x_raw = ground_velocity - np.dot(ground_velocity, z_axis) * z_axis
    x_norm = np.linalg.norm(x_raw)
    if x_norm < 1e-12:
        x_raw = radial - np.dot(radial, z_axis) * z_axis
        x_norm = np.linalg.norm(x_raw)
    if x_norm < 1e-12:
        raise ValueError("coordinate FMC roll reference is degenerate")
    x_axis = x_raw / x_norm
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)

    inertial_to_body = np.empty((3, 3))
    inertial_to_body[0, :] = x_axis
    inertial_to_body[1, :] = y_axis
    inertial_to_body[2, :] = z_axis
    return inertial_to_body


@njit(cache=True)
def _coordinate_reference(state, epoch_jd, target_ecef, mu):
    """Return full frame attitude and feed-forward body rate.

    The body +Z axis points from the spacecraft to the target. Body +X follows
    projected velocity to resolve roll. A central difference of the complete
    frame includes both line-of-sight motion and roll-frame motion.
    """
    position = state[7:10]
    velocity = state[10:13]
    dcm = _coordinate_dcm(position, velocity, epoch_jd, target_ecef)
    quaternion = dcm_to_quaternion(dcm)

    # Differentiate the complete frame over a one-second centered interval.
    # The central-body two-body acceleration provides the state derivative;
    # Earth rotation enters through the epoch-dependent ECEF/GCRF transform.
    radius = np.linalg.norm(position)
    if radius == 0.0:
        raise ValueError("coordinate goal is undefined at zero position")
    acceleration = -mu * position / radius**3
    step = 1.0
    plus_position = position + step * velocity + 0.5 * step * step * acceleration
    minus_position = position - step * velocity + 0.5 * step * step * acceleration
    plus_velocity = velocity + step * acceleration
    minus_velocity = velocity - step * acceleration
    day_step = step / 86400.0
    dcm_plus = _coordinate_dcm(
        plus_position, plus_velocity, epoch_jd + day_step, target_ecef
    )
    dcm_minus = _coordinate_dcm(
        minus_position, minus_velocity, epoch_jd - day_step, target_ecef
    )
    dcm_dot = (dcm_plus - dcm_minus) / (2.0 * step)
    omega_skew_body = -dcm_dot @ dcm.T
    body_rate = np.array(
        [
            0.5 * (omega_skew_body[2, 1] - omega_skew_body[1, 2]),
            0.5 * (omega_skew_body[0, 2] - omega_skew_body[2, 0]),
            0.5 * (omega_skew_body[1, 0] - omega_skew_body[0, 1]),
        ]
    )
    return quaternion, body_rate


@njit(cache=True)
def _coordinate_fmc_rate(state, epoch_jd, target_ecef, mu, fmc_level):
    """Return full attitude/rate for one FMC level and geodetic coordinate."""
    position = state[7:10]
    velocity = state[10:13]
    dcm = _coordinate_fmc_dcm(
        position, velocity, epoch_jd, target_ecef, fmc_level
    )
    quaternion = dcm_to_quaternion(dcm)

    radius = np.linalg.norm(position)
    if radius == 0.0:
        raise ValueError("coordinate FMC goal is undefined at zero position")
    acceleration = -mu * position / radius**3
    step = 1.0
    plus_position = position + step * velocity + 0.5 * step * step * acceleration
    minus_position = position - step * velocity + 0.5 * step * step * acceleration
    plus_velocity = velocity + step * acceleration
    minus_velocity = velocity - step * acceleration
    day_step = step / 86400.0
    dcm_plus = _coordinate_fmc_dcm(
        plus_position, plus_velocity, epoch_jd + day_step, target_ecef, fmc_level
    )
    dcm_minus = _coordinate_fmc_dcm(
        minus_position, minus_velocity, epoch_jd - day_step, target_ecef, fmc_level
    )
    dcm_dot = (dcm_plus - dcm_minus) / (2.0 * step)
    omega_skew_body = -dcm_dot @ dcm.T
    body_rate = np.array(
        [
            0.5 * (omega_skew_body[2, 1] - omega_skew_body[1, 2]),
            0.5 * (omega_skew_body[0, 2] - omega_skew_body[2, 0]),
            0.5 * (omega_skew_body[1, 0] - omega_skew_body[0, 1]),
        ]
    )
    return quaternion, body_rate


class CoordinateGoal(Goal):
    """Track a fixed WGS84 latitude/longitude/altitude coordinate.

    Parameters
    ----------
    lat : float
        WGS84 geodetic latitude in degrees.
    lon : float
        WGS84 longitude in degrees.
    alt : float
        Altitude above the WGS84 ellipsoid in kilometres, matching the
        upstream ``Coordinate_Goal`` constructor.
    """

    kernel = staticmethod(_coordinate_reference)

    def __init__(self, lat: float, lon: float, alt: float = 0.0) -> None:
        lat, lon, alt = float(lat), float(lon), float(alt)
        if not all(map(math.isfinite, (lat, lon, alt))):
            raise ValueError("latitude, longitude, and altitude must be finite")
        if not -90.0 <= lat <= 90.0:
            raise ValueError("latitude must be between -90 and 90 degrees")
        if not -180.0 <= lon <= 180.0:
            raise ValueError("longitude must be between -180 and 180 degrees")
        if alt < -6378.137:
            raise ValueError("altitude is below the WGS84 ellipsoid center")
        self.lat_deg = lat
        self.lon_deg = lon
        self.alt_km = alt
        self.target_ecef_m = self._geodetic_to_ecef(lat, lon, alt)

    @staticmethod
    def _geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_km: float) -> np.ndarray:
        lat = math.radians(lat_deg)
        lon = math.radians(lon_deg)
        semi_major = 6378137.0
        flattening = 1.0 / 298.257223563
        eccentricity_sq = flattening * (2.0 - flattening)
        altitude_m = alt_km * 1000.0
        prime_vertical = semi_major / math.sqrt(
            1.0 - eccentricity_sq * math.sin(lat) ** 2
        )
        return np.array(
            [
                (prime_vertical + altitude_m) * math.cos(lat) * math.cos(lon),
                (prime_vertical + altitude_m) * math.cos(lat) * math.sin(lon),
                (prime_vertical * (1.0 - eccentricity_sq) + altitude_m)
                * math.sin(lat),
            ],
            dtype=np.float64,
        )

    def reference(
        self, state: np.ndarray, epoch_jd: float, runtime: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        state = validate_reference_state(state)
        if not math.isfinite(float(epoch_jd)):
            raise ValueError("epoch_jd must be finite")
        return self.kernel(state, float(epoch_jd), self.target_ecef_m, runtime.mu)


coordinategoal = CoordinateGoal
