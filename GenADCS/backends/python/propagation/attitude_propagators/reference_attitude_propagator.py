"""Prescribed body attitude from orbit or Sun geometry."""

import math

import numpy as np
from numba import njit

from GenADCS.backends.python.environment.ephemeris.ephemeris_model import (
    planetary_state,
)
from GenADCS.backends.python.helpers.math.rotations.quaternion.conversions import dcm_to_quaternion
from GenADCS.backends.python.helpers.math.time.conversions import utc_jd_to_tdb_jd

_MODES = {"nadir": 0, "nadir-pointing": 0, "along-track": 1, "sun": 2, "sun-pointing": 2}


@njit(cache=True)
def _unit_and_rate(vector, derivative):
    length = np.linalg.norm(vector)
    if length == 0.0:
        raise ValueError("reference direction is undefined")
    unit = vector / length
    return unit, (derivative - np.dot(unit, derivative) * unit) / length


@njit(cache=True)
def set_reference_attitude(y, epoch_jd, mode, runtime):
    """Set body-from-inertial quaternion and body rate from moving frame axes."""
    position = y[7:10]
    velocity = y[10:13]
    radius = np.linalg.norm(position)
    acceleration = -runtime.mu * position / radius**3
    radial, radial_dot = _unit_and_rate(position, velocity)
    tangent_raw = velocity - np.dot(velocity, radial) * radial
    tangent_dot_raw = (
        acceleration
        - (np.dot(acceleration, radial) + np.dot(velocity, radial_dot)) * radial
        - np.dot(velocity, radial) * radial_dot
    )
    tangent, tangent_dot = _unit_and_rate(tangent_raw, tangent_dot_raw)

    if mode == 0:
        z, z_dot = -radial, -radial_dot
        x_raw, x_dot_raw = tangent, tangent_dot
    elif mode == 1:
        z, z_dot = tangent, tangent_dot
        x_raw, x_dot_raw = radial, radial_dot
    else:
        tdb_jd = utc_jd_to_tdb_jd(epoch_jd)
        sun, sun_velocity = planetary_state(
            10, tdb_jd, runtime.ephemeris_segments,
            runtime.ephemeris_times, runtime.ephemeris_data,
        )
        sun *= 1000.0
        sun_velocity *= 1000.0
        z, z_dot = _unit_and_rate(sun - position, sun_velocity - velocity)
        reference, reference_dot = tangent, tangent_dot
        projection = reference - np.dot(reference, z) * z
        if np.linalg.norm(projection) < 1e-10:
            reference, reference_dot = radial, radial_dot
        scalar = np.dot(reference, z)
        scalar_dot = np.dot(reference_dot, z) + np.dot(reference, z_dot)
        x_raw = reference - scalar * z
        x_dot_raw = reference_dot - scalar_dot * z - scalar * z_dot

    x, x_dot = _unit_and_rate(x_raw, x_dot_raw)
    y_raw = np.cross(z, x)
    y_dot_raw = np.cross(z_dot, x) + np.cross(z, x_dot)
    body_y, body_y_dot = _unit_and_rate(y_raw, y_dot_raw)
    # Reorthogonalize x in the same way as the quaternion construction.
    body_x = np.cross(body_y, z)
    body_x_dot = np.cross(body_y_dot, z) + np.cross(body_y, z_dot)
    matrix = np.empty((3, 3))
    matrix[0, :] = body_x
    matrix[1, :] = body_y
    matrix[2, :] = z
    q = dcm_to_quaternion(matrix)
    y[:4] = q
    y[4] = -np.dot(body_y_dot, z)
    y[5] = -np.dot(z_dot, body_x)
    y[6] = -np.dot(body_x_dot, body_y)


class ReferenceAttitudePropagator:
    """Prescribed nadir, along-track, or Sun pointing for a Keplerian orbit."""

    def __init__(self, dt: float, mode: str) -> None:
        self.dt = float(dt)
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be finite and positive seconds")
        try:
            self.mode = _MODES[mode]
        except KeyError as exc:
            raise ValueError(f"unknown reference attitude mode: {mode!r}") from exc
