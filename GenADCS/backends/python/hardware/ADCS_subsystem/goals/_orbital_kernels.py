"""Numba-compiled inertial and orbit-relative attitude goals."""

from __future__ import annotations


import numpy as np
from numba import njit

from GenADCS.backends.python.environment.ephemeris.ephemeris_model import planetary_state
from GenADCS.backends.python.helpers.math.rotations.quaternion.conversions import (
    dcm_to_quaternion,
)
from GenADCS.backends.python.helpers.math.time.conversions import utc_jd_to_tdb_jd


@njit(cache=True, inline="always")
def _unit_rate(vector, derivative):
    length = np.linalg.norm(vector)
    if length <= 0.0:
        raise ValueError("reference direction is undefined")
    unit = vector / length
    return unit, (derivative - np.dot(unit, derivative) * unit) / length


@njit(cache=True)
def _reference(state, epoch_jd, runtime, mode):
    """Construct attitude/rate using the propagator's moving-frame convention.

    Body +Z is the primary pointing axis. The selected secondary direction
    fixes roll. Mode IDs: nadir, zenith, along-track, cross-track, velocity,
    Sun. The returned body rate is resolved in body coordinates.
    """
    position = state[7:10]
    velocity = state[10:13]
    radius = np.linalg.norm(position)
    if radius <= 0.0:
        raise ValueError("orbital frame is undefined at zero position")
    acceleration = -runtime.mu * position / radius**3
    radial, radial_dot = _unit_rate(position, velocity)

    tangent_raw = velocity - np.dot(velocity, radial) * radial
    tangent_dot_raw = (
        acceleration
        - (np.dot(acceleration, radial) + np.dot(velocity, radial_dot)) * radial
        - np.dot(velocity, radial) * radial_dot
    )
    tangent, tangent_dot = _unit_rate(tangent_raw, tangent_dot_raw)
    orbit_normal_raw = np.cross(position, velocity)
    orbit_normal_dot_raw = np.cross(position, acceleration)
    orbit_normal, orbit_normal_dot = _unit_rate(
        orbit_normal_raw, orbit_normal_dot_raw
    )

    if mode == 0:  # nadir
        z, z_dot = -radial, -radial_dot
        x_raw, x_dot_raw = tangent, tangent_dot
    elif mode == 1:  # zenith
        z, z_dot = radial, radial_dot
        x_raw, x_dot_raw = tangent, tangent_dot
    elif mode == 2:  # along-track
        z, z_dot = tangent, tangent_dot
        x_raw, x_dot_raw = radial, radial_dot
    elif mode == 3:  # cross-track
        z, z_dot = orbit_normal, orbit_normal_dot
        x_raw, x_dot_raw = radial, radial_dot
    elif mode == 4:  # instantaneous velocity
        z, z_dot = _unit_rate(velocity, acceleration)
        x_raw, x_dot_raw = radial, radial_dot
    else:  # Sun-pointing
        tdb_jd = utc_jd_to_tdb_jd(epoch_jd)
        sun, sun_velocity = planetary_state(
            10,
            tdb_jd,
            runtime.ephemeris_segments,
            runtime.ephemeris_times,
            runtime.ephemeris_data,
        )
        sun *= 1000.0
        sun_velocity *= 1000.0
        z, z_dot = _unit_rate(sun - position, sun_velocity - velocity)
        x_raw, x_dot_raw = tangent, tangent_dot
        projected = x_raw - np.dot(x_raw, z) * z
        if np.linalg.norm(projected) < 1e-10:
            x_raw, x_dot_raw = radial, radial_dot

    # Project the secondary axis into the plane normal to body +Z and include
    # the derivative of the projection so the reference rate tracks the frame.
    scalar = np.dot(x_raw, z)
    scalar_dot = np.dot(x_dot_raw, z) + np.dot(x_raw, z_dot)
    x_projected = x_raw - scalar * z
    x_projected_dot = x_dot_raw - scalar_dot * z - scalar * z_dot
    x, x_dot = _unit_rate(x_projected, x_projected_dot)
    y_raw = np.cross(z, x)
    y_dot_raw = np.cross(z_dot, x) + np.cross(z, x_dot)
    body_y, body_y_dot = _unit_rate(y_raw, y_dot_raw)
    body_x = np.cross(body_y, z)
    body_x_dot = np.cross(body_y_dot, z) + np.cross(body_y, z_dot)

    dcm = np.empty((3, 3))
    dcm[0, :] = body_x
    dcm[1, :] = body_y
    dcm[2, :] = z
    quaternion = dcm_to_quaternion(dcm)
    rate = np.empty(3)
    rate[0] = -np.dot(body_y_dot, z)
    rate[1] = -np.dot(z_dot, body_x)
    rate[2] = -np.dot(body_x_dot, body_y)
    return quaternion, rate


@njit(cache=True)
def _nadir(state, epoch_jd, runtime):
    return _reference(state, epoch_jd, runtime, 0)


@njit(cache=True)
def _zenith(state, epoch_jd, runtime):
    return _reference(state, epoch_jd, runtime, 1)


@njit(cache=True)
def _alongtrack(state, epoch_jd, runtime):
    return _reference(state, epoch_jd, runtime, 2)


@njit(cache=True)
def _crosstrack(state, epoch_jd, runtime):
    return _reference(state, epoch_jd, runtime, 3)


@njit(cache=True)
def _velocity(state, epoch_jd, runtime):
    return _reference(state, epoch_jd, runtime, 4)


@njit(cache=True)
def _sun(state, epoch_jd, runtime):
    return _reference(state, epoch_jd, runtime, 5)
