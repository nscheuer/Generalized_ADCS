"""Shared Numba SPK evaluation kernels and ephemeris model interface."""

import numpy as np
from numba import njit

BODY_SUN = 0
BODY_EARTH = 1
BODY_MOON = 2
_SECONDS_PER_DAY = 86400.0
_J2000 = 2451545.0
_EARTH_TARGET = 399
_TABLE_TARGETS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 301, 10, 399], dtype=np.int64)


@njit(cache=True)
def _segment_state(target, et, segments, times, data):
    """Evaluate an SPK type 2/3 segment; output is km and km/s."""
    selected = -1
    for i in range(segments.shape[0]):
        if segments[i, 0] == target and times[i, 0] <= et <= times[i, 1]:
            selected = i
    if selected < 0:
        raise ValueError("no SPK segment covers the requested target and epoch")

    offset = segments[selected, 3]
    trailer = offset + segments[selected, 4] - 4
    segment_type = segments[selected, 2]
    initial = data[trailer]
    interval_length = data[trailer + 1]
    record_size = int(data[trailer + 2])
    record_count = int(data[trailer + 3])
    record = int((et - initial) / interval_length)
    if record >= record_count:
        record = record_count - 1
    if record < 0:
        record = 0
    base = offset + record * record_size
    midpoint = data[base]
    radius = data[base + 1]
    x = (et - midpoint) / radius

    if segment_type == 2:
        ncf = (record_size - 2) // 3
    else:
        ncf = (record_size - 2) // 6
    position = np.zeros(3, dtype=np.float64)
    velocity = np.zeros(3, dtype=np.float64)
    for axis in range(3):
        p_offset = base + 2 + axis * ncf
        v_offset = base + 2 + (axis + 3) * ncf if segment_type == 3 else -1
        t0 = 1.0
        t1 = x
        d0 = 0.0
        d1 = 1.0
        position[axis] = data[p_offset]
        if ncf > 1:
            position[axis] += data[p_offset + 1] * x
            if segment_type == 3:
                velocity[axis] = data[v_offset] + data[v_offset + 1] * x
        for k in range(2, ncf):
            t2 = 2.0 * x * t1 - t0
            d2 = 2.0 * t1 + 2.0 * x * d1 - d0
            position[axis] += data[p_offset + k] * t2
            if segment_type == 3:
                velocity[axis] += data[v_offset + k] * t2
            t0, t1 = t1, t2
            d0, d1 = d1, d2
        if segment_type == 2:
            velocity[axis] = data[p_offset + 1] if ncf == 1 else 0.0
            if ncf > 1:
                # derivative of the position polynomial, evaluated by recurrence
                t0 = 1.0
                t1 = x
                d0 = 0.0
                d1 = 1.0
                velocity[axis] = data[p_offset + 1]
                for k in range(2, ncf):
                    t2 = 2.0 * x * t1 - t0
                    d2 = 2.0 * t1 + 2.0 * x * d1 - d0
                    velocity[axis] += data[p_offset + k] * d2
                    t0, t1 = t1, t2
                    d0, d1 = d1, d2
            velocity[axis] *= 1.0 / radius
    return position, velocity, segments[selected, 1]


@njit(cache=True)
def _barycentric_state(target, et, segments, times, data):
    pos = np.zeros(3, dtype=np.float64)
    vel = np.zeros(3, dtype=np.float64)
    node = target
    for _ in range(8):
        if node == 0:
            return pos, vel
        rel_pos, rel_vel, center = _segment_state(node, et, segments, times, data)
        pos += rel_pos
        vel += rel_vel
        node = center
    raise ValueError("SPK center chain is too deep")


@njit(cache=True)
def planetary_position(body_index, epoch_jd, segments, times, data):
    """Earth-centered position [km] for a DE421 table body index (0-11)."""
    if body_index < 0 or body_index >= _TABLE_TARGETS.size:
        raise ValueError("body_index must be in the DE421 body range 0..11")
    et = (epoch_jd - _J2000) * _SECONDS_PER_DAY
    target = _TABLE_TARGETS[body_index]
    if target == _EARTH_TARGET:
        return np.zeros(3, dtype=np.float64)
    target_pos, _ = _barycentric_state(target, et, segments, times, data)
    earth_pos, _ = _barycentric_state(_EARTH_TARGET, et, segments, times, data)
    return target_pos - earth_pos


@njit(cache=True)
def planetary_velocity(body_index, epoch_jd, segments, times, data):
    """Earth-centered velocity [km/s] for a DE421 table body index (0-11)."""
    if body_index < 0 or body_index >= _TABLE_TARGETS.size:
        raise ValueError("body_index must be in the DE421 body range 0..11")
    et = (epoch_jd - _J2000) * _SECONDS_PER_DAY
    target = _TABLE_TARGETS[body_index]
    if target == _EARTH_TARGET:
        return np.zeros(3, dtype=np.float64)
    _, target_vel = _barycentric_state(target, et, segments, times, data)
    _, earth_vel = _barycentric_state(_EARTH_TARGET, et, segments, times, data)
    return target_vel - earth_vel


@njit(cache=True)
def planetary_state(body_index, epoch_jd, segments, times, data):
    """Earth-centered position and velocity from one SPK traversal [km, km/s]."""
    if body_index < 0 or body_index >= _TABLE_TARGETS.size:
        raise ValueError("body_index must be in the DE421 body range 0..11")
    target = _TABLE_TARGETS[body_index]
    if target == _EARTH_TARGET:
        return np.zeros(3), np.zeros(3)
    et = (epoch_jd - _J2000) * _SECONDS_PER_DAY
    target_pos, target_vel = _barycentric_state(target, et, segments, times, data)
    earth_pos, earth_vel = _barycentric_state(_EARTH_TARGET, et, segments, times, data)
    return target_pos - earth_pos, target_vel - earth_vel


@njit(cache=True)
def ephemeris_position(body_id, epoch_jd, segments, times, data):
    """Earth-centered DE421 position [km]; epoch is TDB Julian date."""
    if body_id == BODY_EARTH:
        return np.zeros(3, dtype=np.float64)
    if body_id == BODY_SUN:
        index = 10
    elif body_id == BODY_MOON:
        index = 9
    else:
        raise ValueError("body_id must be BODY_SUN, BODY_EARTH, or BODY_MOON")
    return planetary_position(index, epoch_jd, segments, times, data)


@njit(cache=True)
def ephemeris_velocity(body_id, epoch_jd, segments, times, data):
    """Earth-centered DE421 velocity [km/s]; epoch is TDB Julian date."""
    if body_id == BODY_EARTH:
        return np.zeros(3, dtype=np.float64)
    if body_id == BODY_SUN:
        index = 10
    elif body_id == BODY_MOON:
        index = 9
    else:
        raise ValueError("body_id must be BODY_SUN, BODY_EARTH, or BODY_MOON")
    return planetary_velocity(index, epoch_jd, segments, times, data)


@njit(cache=True)
def eclipse_state(spacecraft_position_m, sun_position_m, occulting_body_radius_m, sun_radius_m):
    spacecraft_to_sun = sun_position_m - spacecraft_position_m
    spacecraft_to_body = -spacecraft_position_m
    sun_distance = np.sqrt(np.dot(spacecraft_to_sun, spacecraft_to_sun))
    body_distance = np.sqrt(np.dot(spacecraft_to_body, spacecraft_to_body))
    if sun_distance <= sun_radius_m or body_distance <= occulting_body_radius_m:
        return 0.0, True, False
    sun_angle = np.arcsin(min(1.0, sun_radius_m / sun_distance))
    body_angle = np.arcsin(min(1.0, occulting_body_radius_m / body_distance))
    cosine_separation = np.dot(spacecraft_to_sun, spacecraft_to_body) / (sun_distance * body_distance)
    separation = np.arccos(min(1.0, max(-1.0, cosine_separation)))
    if separation >= sun_angle + body_angle:
        return 1.0, False, False
    if body_angle >= sun_angle and separation <= body_angle - sun_angle:
        return 0.0, True, False
    if separation <= abs(body_angle - sun_angle):
        overlap_area = np.pi * min(body_angle, sun_angle) ** 2
    else:
        sun_term = (separation * separation + sun_angle * sun_angle - body_angle * body_angle) / (2.0 * separation * sun_angle)
        body_term = (separation * separation + body_angle * body_angle - sun_angle * sun_angle) / (2.0 * separation * body_angle)
        radicand = (-separation + sun_angle + body_angle) * (separation + sun_angle - body_angle)
        radicand *= (separation - sun_angle + body_angle) * (separation + sun_angle + body_angle)
        overlap_area = (sun_angle * sun_angle * np.arccos(min(1.0, max(-1.0, sun_term)))
                        + body_angle * body_angle * np.arccos(min(1.0, max(-1.0, body_term)))
                        - 0.5 * np.sqrt(max(0.0, radicand)))
    illumination = min(1.0, max(0.0, 1.0 - overlap_area / (np.pi * sun_angle * sun_angle)))
    return illumination, False, True


class EphemerisModel:
    """Base class for models backed by parsed SPK segments."""

    def __init__(self, segments, times, data):
        self._segments = np.ascontiguousarray(segments, dtype=np.int64)
        self._times = np.ascontiguousarray(times, dtype=np.float64)
        self._data = np.ascontiguousarray(data, dtype=np.float64)

    def position(self, body_id, epoch_jd):
        return ephemeris_position(int(body_id), float(epoch_jd), self._segments, self._times, self._data)

    def velocity(self, body_id, epoch_jd):
        return ephemeris_velocity(int(body_id), float(epoch_jd), self._segments, self._times, self._data)

    @staticmethod
    def eclipse_state(spacecraft_position_m, sun_position_m, occulting_body_radius_m, sun_radius_m):
        return eclipse_state(spacecraft_position_m, sun_position_m, occulting_body_radius_m, sun_radius_m)
