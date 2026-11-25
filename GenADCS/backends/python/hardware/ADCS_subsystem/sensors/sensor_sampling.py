"""Prepared sensor configuration and compiled sampling at subsystem ticks."""

import numpy as np
from numba import njit
from GenADCS.backends.python.environment import Environment

from GenADCS.backends.python.dynamics.srp_model.projected_area import SUN_RADIUS_M
from GenADCS.backends.python.environment.ephemeris import (
    BODY_SUN,
    eclipse_state,
    ephemeris_position,
)
from GenADCS.backends.python.environment.gravity.j_gravity_model import (
    gravity_acceleration_inertial_m_s2,
)
from GenADCS.backends.python.environment.magnetic_field_model.chaos8.chaos8 import (
    chaos8_field_inertial_tesla,
)
from GenADCS.backends.python.environment.magnetic_field_model.igrf14.igrf14 import (
    igrf14_field_inertial_tesla,
)
from GenADCS.backends.python.helpers.math.time.conversions import utc_jd_to_tdb_jd
from GenADCS.backends.python.helpers.random.philox import standard_normals
from GenADCS.backends.python.propagation.coupled_propagators._coupled_dynamics import (
    _derivative,
)

from .kernels import (
    fine_direction_available,
    fine_direction_measurement,
    gnss_measurement,
    rate_measurement,
    rotate_inertial_to_sensor,
    scalar_sun_measurement,
    star_tracker_measurement,
    vector_measurement,
)

DEFAULT_SEED = np.uint64(0)

SENSOR_NAMES = (
    "gyroscope",
    "magnetometer",
    "autonomous_star_tracker",
    "coarse_sun_sensor",
    "fine_sun_sensor",
    "fine_earth_sensor",
    "gnss",
    "accelerometer",
)
SENSOR_CLASSES = (
    "Gyroscope",
    "Magnetometer",
    "AutonomousStarTracker",
    "CoarseSunSensor",
    "FineSunSensor",
    "FineEarthSensor",
    "GNSS",
    "Accelerometer",
)
WIDTHS = (3, 3, 4, 1, 2, 2, 6, 3)


def prepare_sensors(
    sensors: list, environment: Environment
) -> tuple[np.ndarray, ...]:
    """Convert heterogeneous Python sensor objects to fixed numeric arrays once."""
    n = len(sensors)
    code = np.empty(n, np.int64)
    orientation = np.zeros((n, 4))
    misalignment = np.zeros((n, 4))
    bias = np.zeros((n, 6))
    chol = np.zeros((n, 6, 6))
    normal = np.zeros((n, 3))
    fov = np.zeros((n, 2))
    scalars = np.zeros((n, 3))  # saturation, sensitivity, noise std
    enabled = np.empty(n, np.bool_)
    offsets = np.empty(n, np.int64)
    counts = np.zeros(len(SENSOR_NAMES), np.int64)
    for i, sensor in enumerate(sensors):
        try:
            kind = SENSOR_CLASSES.index(type(sensor).__name__)
        except ValueError as exc:
            raise TypeError(
                f"unsupported sensor type: {type(sensor).__name__}"
            ) from exc
        code[i] = kind
        offsets[i] = counts[kind]
        counts[kind] += WIDTHS[kind]
        enabled[i] = sensor.enabled
        if kind != 6:
            orientation[i] = sensor.orientation
            misalignment[i] = sensor.misalignment
        if kind == 3:
            normal[i] = sensor.normal
            scalars[i] = sensor.saturation_value, sensor.sensitivity, sensor.noise_std
            bias[i, 0] = sensor.bias
            fov[i, 0] = sensor.fov_half_angle
        elif kind == 2:
            fov[i, 0] = sensor.fov_half_angle
            noise = sensor.noise_cholesky
            chol[i, :3, :3] = noise
        else:
            if kind != 2:
                bias[i, : WIDTHS[kind]] = sensor.bias
            if kind in (0, 1, 7):
                scalars[i, 0] = sensor.saturation_value
            if kind in (4, 5):
                fov[i] = sensor.fov_half_angles
            noise = sensor.noise_cholesky
            chol[i, : noise.shape[0], : noise.shape[1]] = noise
    if np.any(code == 1) and environment.magnetic_field is None:
        raise ValueError("magnetometer requires an environment magnetic model")
    if (
        np.any((code == 2) | (code == 3) | (code == 4))
        and environment.ephemeris is None
    ):
        raise ValueError("Sun and star tracker sensors require an ephemeris")
    starts = np.zeros(len(SENSOR_NAMES), np.int64)
    starts[1:] = np.cumsum(counts[:-1])
    for i in range(n):
        offsets[i] += starts[code[i]]
    return (
        code,
        orientation,
        misalignment,
        bias,
        chol,
        normal,
        fov,
        scalars,
        enabled,
        offsets,
        counts,
        starts,
    )


@njit(cache=True)
def sample_sensors(
    y: np.ndarray,
    jd: float,
    p: object,
    config: tuple,
    clean: np.ndarray,
    reading: np.ndarray,
    seed: int = DEFAULT_SEED,
    tick: int = 0,
) -> None:
    """Write every sensor's clean and noisy values into preallocated rows."""
    (
        code,
        orientation,
        misalignment,
        bias,
        chol,
        normal,
        fov,
        scalars,
        enabled,
        offsets,
        _counts,
        _starts,
    ) = config
    q, w, pos, vel = y[:4], y[4:7], y[7:10], y[10:13]
    sun = np.zeros(3)
    illumination = 0.0
    field = np.zeros(3)
    specific = np.zeros(3)
    if np.any((code == 2) | (code == 3) | (code == 4)) and p.ephemeris_segments.size:
        sun_position = 1000.0 * ephemeris_position(
            BODY_SUN,
            utc_jd_to_tdb_jd(jd),
            p.ephemeris_segments,
            p.ephemeris_times,
            p.ephemeris_data,
        )
        sun = sun_position - pos
        illumination, _, _ = eclipse_state(pos, sun_position, p.radius, SUN_RADIUS_M)
    if np.any(code == 1):
        if p.magnetic_kind == 1:
            field = igrf14_field_inertial_tesla(
                pos, jd, p.magnetic_times, p.magnetic_g, p.magnetic_h
            )
        else:
            field = chaos8_field_inertial_tesla(
                pos,
                jd,
                p.magnetic_times,
                p.magnetic_g,
                p.magnetic_h,
                p.magnetic_static_g,
                p.magnetic_static_h,
                p.magnetic_include_static,
            )
    if np.any(code == 7):
        gravity = gravity_acceleration_inertial_m_s2(pos, jd, p.mu, p.radius, p.zonals)
        specific = _derivative(y, jd, p)[10:13] - gravity
    for i in range(code.size):
        kind = code[i]
        offset = offsets[i]
        if not enabled[i]:
            continue
        ori = orientation[i]
        mis = misalignment[i]
        b = bias[i]
        c = chol[i]
        s = scalars[i]
        draw = standard_normals(
            seed,
            tick,
            i,
            3
            if kind == 2
            else (1 if kind == 3 else (2 if kind == 4 or kind == 5 else WIDTHS[kind])),
        )
        if kind == 0:
            a = rate_measurement(w, ori, mis, b[:3], c[:3, :3], s[0], False, False)
            v = rate_measurement(w, ori, mis, b[:3], c[:3, :3], s[0], True, True, draw)
        elif kind == 1:
            a = vector_measurement(
                field, q, ori, mis, b[:3], c[:3, :3], s[0], False, False
            )
            v = vector_measurement(
                field, q, ori, mis, b[:3], c[:3, :3], s[0], True, True, draw
            )
        elif kind == 2:
            a = star_tracker_measurement(q, ori, mis, c[:3, :3], False)
            earth = -pos
            earth_sensor = rotate_inertial_to_sensor(earth, q, ori, mis)
            sun_sensor = (
                rotate_inertial_to_sensor(sun, q, ori, mis)
                if p.ephemeris_segments.size
                else np.zeros(3)
            )
            earth_occulted = (
                np.dot(earth_sensor, earth_sensor) > 0.0
                and earth_sensor[2] > 0.0
                and np.arctan2(
                    np.sqrt(earth_sensor[0] ** 2 + earth_sensor[1] ** 2),
                    earth_sensor[2],
                )
                < fov[i, 0]
            )
            sun_occulted = (
                np.dot(sun_sensor, sun_sensor) > 0.0
                and sun_sensor[2] > 0.0
                and np.arctan2(
                    np.sqrt(sun_sensor[0] ** 2 + sun_sensor[1] ** 2), sun_sensor[2]
                )
                < fov[i, 0]
            )
            if earth_occulted or sun_occulted:
                continue
            v = star_tracker_measurement(q, ori, mis, c[:3, :3], True, draw)
        elif kind == 3:
            if illumination <= 0.0:
                clean[offset] = np.nan
                reading[offset] = np.nan
                continue
            a = np.array(
                [
                    scalar_sun_measurement(
                        sun,
                        q,
                        ori,
                        mis,
                        normal[i],
                        s[1],
                        b[0],
                        s[2],
                        s[0],
                        False,
                        False,
                    )
                ]
            )
            v = np.array(
                [
                    scalar_sun_measurement(
                        sun,
                        q,
                        ori,
                        mis,
                        normal[i],
                        s[1],
                        b[0],
                        s[2],
                        s[0],
                        True,
                        True,
                        draw,
                    )
                ]
            )
        elif kind == 4 or kind == 5:
            direction = sun if kind == 4 else -pos
            if not fine_direction_available(
                direction,
                q,
                ori,
                mis,
                fov[i],
                True,
                illumination > 0 if kind == 4 else True,
            ):
                continue
            a = fine_direction_measurement(
                direction, q, ori, mis, b[:2], c[:2, :2], False, False
            )
            v = fine_direction_measurement(
                direction, q, ori, mis, b[:2], c[:2, :2], True, True, draw
            )
        elif kind == 6:
            a = gnss_measurement(pos, vel, b, c, False, False)
            v = gnss_measurement(pos, vel, b, c, True, True, draw)
        else:
            a = vector_measurement(
                specific, q, ori, mis, b[:3], c[:3, :3], s[0], False, False
            )
            v = vector_measurement(
                specific, q, ori, mis, b[:3], c[:3, :3], s[0], True, True, draw
            )
        clean[offset : offset + a.size] = a
        reading[offset : offset + v.size] = v
