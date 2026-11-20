import numpy as np
from numba import njit

BODY_SUN = 0
BODY_EARTH = 1
BODY_MOON = 2


@njit(cache=True)
def ephemeris_position(body_id, epoch_jd):
    raise NotImplementedError


@njit(cache=True)
def ephemeris_velocity(body_id, epoch_jd):
    raise NotImplementedError


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
        overlap_area = (
            sun_angle * sun_angle * np.arccos(min(1.0, max(-1.0, sun_term)))
            + body_angle * body_angle * np.arccos(min(1.0, max(-1.0, body_term)))
            - 0.5 * np.sqrt(max(0.0, radicand))
        )

    illumination = min(1.0, max(0.0, 1.0 - overlap_area / (np.pi * sun_angle * sun_angle)))
    return illumination, False, True


class EphemerisModel:
    position = staticmethod(ephemeris_position)
    velocity = staticmethod(ephemeris_velocity)
    eclipse_state = staticmethod(eclipse_state)
