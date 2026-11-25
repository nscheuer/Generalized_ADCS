"""Self-contained rotations between GCRF, ECEF, Body, LVLH, and TEME.

Matrices map column vectors from source to target. Earth orientation uses a
low-order IAU 1976 precession / four-term nutation model and GMST82, with
optional polar motion (xp, yp in radians). Supply Julian dates on UT1 and TT
scales for Earth-fixed or TEME transforms. This compact model omits higher-order
celestial terms and sub-daily Earth-orientation corrections.

LVLH axes are +x radial, +y along-track, +z orbit-normal. Body attitude uses
the scalar-first Hamilton quaternion q_body_from_gcrf.
"""

import numpy as np
from numba import njit

from ..rotations.quaternion.conversions import _quaternion_to_dcm

_FRAME_NAMES = {"GCRF", "ECEF", "BODY", "LVLH", "TEME"}
_ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)


@njit(cache=True)
def _r1(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)))


@njit(cache=True)
def _r2(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)))


@njit(cache=True)
def _r3(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


@njit(cache=True)
def _precession_gcrf_to_mod(jd_tt):
    t = (jd_tt - 2451545.0) / 36525.0
    arcsec = np.pi / (180.0 * 3600.0)
    zeta = (2306.2181 * t + 0.30188 * t * t + 0.017998 * t**3) * arcsec
    z = (2306.2181 * t + 1.09468 * t * t + 0.018203 * t**3) * arcsec
    theta = (2004.3109 * t - 0.42665 * t * t - 0.041833 * t**3) * arcsec
    return _r3(-z) @ _r2(theta) @ _r3(-zeta)


@njit(cache=True)
def _nutation_4term(jd_tt):
    t = (jd_tt - 2451545.0) / 36525.0
    deg = np.pi / 180.0
    arcsec = np.pi / (180.0 * 3600.0)
    omega = (125.04452 - 1934.136261 * t) * deg
    mean_sun = (280.4665 + 36000.7698 * t) * deg
    mean_moon = (218.3165 + 481267.8813 * t) * deg
    dpsi = (
        -17.20 * np.sin(omega)
        - 1.32 * np.sin(2.0 * mean_sun)
        - 0.23 * np.sin(2.0 * mean_moon)
        + 0.21 * np.sin(2.0 * omega)
    ) * arcsec
    deps = (
        9.20 * np.cos(omega)
        + 0.57 * np.cos(2.0 * mean_sun)
        + 0.10 * np.cos(2.0 * mean_moon)
        - 0.09 * np.cos(2.0 * omega)
    ) * arcsec
    eps = (23.43929111 - 0.013004167 * t - 1.64e-7 * t * t + 5.04e-7 * t**3) * deg
    return dpsi, deps, eps


@njit(cache=True)
def _gcrf_to_teme(jd_tt):
    precession = _precession_gcrf_to_mod(jd_tt)
    dpsi, deps, eps = _nutation_4term(jd_tt)
    nutation = _r1(-(eps + deps)) @ _r3(-dpsi) @ _r1(eps)
    # TEME uses the mean equinox with the true equator; remove the equation of
    # equinoxes from the true equator/equinox orientation.
    equation_equinoxes = dpsi * np.cos(eps)
    return _r3(-equation_equinoxes) @ nutation @ precession


@njit(cache=True)
def _gmst82(jd_ut1):
    t = (jd_ut1 - 2451545.0) / 36525.0
    seconds = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * t
        + 0.093104 * t * t
        - 6.2e-6 * t**3
    )
    turns = seconds / 86400.0
    turns -= np.floor(turns)
    return turns * (2.0 * np.pi)


@njit(cache=True)
def _teme_to_ecef(jd_ut1, xp, yp):
    # GMST82 is the conventional TEME-to-PEF rotation; polar motion then maps
    # PEF to the terrestrial frame.
    return _r2(-xp) @ _r1(-yp) @ _r3(-_gmst82(jd_ut1))


@njit(cache=True)
def _gcrf_to_ecef(jd_ut1, jd_tt, xp, yp):
    return _teme_to_ecef(jd_ut1, xp, yp) @ _gcrf_to_teme(jd_tt)


@njit(cache=True)
def gcrf_to_lvlh_matrix(position_gcrf, velocity_gcrf):
    """Return GCRF-to-LVLH rotation; axes are radial, along-track, normal."""
    r = np.asarray(position_gcrf, dtype=np.float64)
    v = np.asarray(velocity_gcrf, dtype=np.float64)
    if r.shape != (3,) or v.shape != (3,):
        raise ValueError("position_gcrf and velocity_gcrf must be 3-vectors")
    r_norm = np.sqrt(np.dot(r, r))
    h = np.cross(r, v)
    h_norm = np.sqrt(np.dot(h, h))
    if r_norm == 0.0 or h_norm == 0.0:
        raise ValueError("LVLH requires nonzero position and angular momentum")
    x_axis = r / r_norm
    z_axis = h / h_norm
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.sqrt(np.dot(y_axis, y_axis))
    return np.vstack((x_axis, y_axis, z_axis))


def _gcrf_to_frame_matrix(
    frame,
    jd_ut1=None,
    jd_tt=None,
    xp=0.0,
    yp=0.0,
    position_gcrf=None,
    velocity_gcrf=None,
    q_body_from_gcrf=None,
):
    if frame == "GCRF":
        return np.eye(3)
    if frame in ("ECEF", "TEME") and jd_ut1 is None:
        raise ValueError(f"jd_ut1 is required for {frame} conversions")
    if frame in ("ECEF", "TEME") and jd_tt is None:
        raise ValueError(f"jd_tt is required for {frame} conversions")
    if frame == "ECEF":
        return _gcrf_to_ecef(float(jd_ut1), float(jd_tt), float(xp), float(yp))
    if frame == "TEME":
        return _gcrf_to_teme(float(jd_tt))
    if frame == "LVLH":
        if position_gcrf is None or velocity_gcrf is None:
            raise ValueError(
                "position_gcrf and velocity_gcrf are required for LVLH conversions"
            )
        return gcrf_to_lvlh_matrix(position_gcrf, velocity_gcrf)
    if frame == "BODY":
        if q_body_from_gcrf is None:
            raise ValueError("q_body_from_gcrf is required for body-frame conversions")
        return np.asarray(
            _quaternion_to_dcm(np.asarray(q_body_from_gcrf, dtype=np.float64))
        )
    raise ValueError(
        f"unsupported frame {frame!r}; expected one of {sorted(_FRAME_NAMES)}"
    )


def frame_rotation(
    source,
    target,
    *,
    jd_ut1=None,
    jd_tt=None,
    xp=0.0,
    yp=0.0,
    position_gcrf=None,
    velocity_gcrf=None,
    q_body_from_gcrf=None,
):
    """Return a 3x3 source-to-target rotation between supported frames.

    ``jd_ut1`` and ``jd_tt`` are required when either frame is ECEF or TEME.
    Polar motion ``xp`` and ``yp`` are in radians. ``position_gcrf`` and
    ``velocity_gcrf`` define LVLH; ``q_body_from_gcrf`` defines Body.
    """
    source = str(source).upper()
    target = str(target).upper()
    if source not in _FRAME_NAMES:
        raise ValueError(f"unsupported source frame {source!r}")
    if target not in _FRAME_NAMES:
        raise ValueError(f"unsupported target frame {target!r}")
    if source == target:
        return np.eye(3)
    r_target_from_gcrf = _gcrf_to_frame_matrix(
        target,
        jd_ut1,
        jd_tt,
        xp,
        yp,
        position_gcrf,
        velocity_gcrf,
        q_body_from_gcrf,
    )
    r_source_from_gcrf = _gcrf_to_frame_matrix(
        source,
        jd_ut1,
        jd_tt,
        xp,
        yp,
        position_gcrf,
        velocity_gcrf,
        q_body_from_gcrf,
    )
    return r_target_from_gcrf @ r_source_from_gcrf.T


def convert_vector(vector, source, target, **frame_context):
    """Rotate a finite 3-vector between frames."""
    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError("vector must be a finite 3-vector")
    return frame_rotation(source, target, **frame_context) @ value


@njit(cache=True)
def gcrf_to_ecef_matrix(jd_ut1, jd_tt, xp=0.0, yp=0.0):
    return _gcrf_to_ecef(float(jd_ut1), float(jd_tt), float(xp), float(yp))


@njit(cache=True)
def ecef_to_gcrf_matrix(jd_ut1, jd_tt, xp=0.0, yp=0.0):
    return gcrf_to_ecef_matrix(jd_ut1, jd_tt, xp, yp).T


def gcrf_to_teme_matrix(jd_tt):
    return _gcrf_to_teme(float(jd_tt))


def teme_to_gcrf_matrix(jd_tt):
    return gcrf_to_teme_matrix(jd_tt).T


def teme_to_ecef_matrix(jd_ut1, xp=0.0, yp=0.0):
    return _teme_to_ecef(float(jd_ut1), float(xp), float(yp))


def ecef_to_teme_matrix(jd_ut1, xp=0.0, yp=0.0):
    return teme_to_ecef_matrix(jd_ut1, xp, yp).T


@njit(cache=True)
def ecef_to_geodetic(position_ecef_m):
    """Convert ECEF metres to WGS84 geodetic latitude/longitude [deg], altitude [km]."""
    x, y, z = position_ecef_m[0], position_ecef_m[1], position_ecef_m[2]
    a = 6378137.0
    e2 = 6.6943799901413165e-3
    p = np.hypot(x, y)
    lon = np.arctan2(y, x)
    if p == 0.0:
        lat = np.pi / 2.0 if z >= 0.0 else -np.pi / 2.0
        altitude = abs(z) - a * np.sqrt(1.0 - e2)
    else:
        lat = np.arctan2(z, p * (1.0 - e2))
        for _ in range(8):
            sin_lat = np.sin(lat)
            prime_vertical_radius = a / np.sqrt(1.0 - e2 * sin_lat * sin_lat)
            lat = np.arctan2(z + e2 * prime_vertical_radius * sin_lat, p)
        sin_lat = np.sin(lat)
        prime_vertical_radius = a / np.sqrt(1.0 - e2 * sin_lat * sin_lat)
        altitude = p / np.cos(lat) - prime_vertical_radius
    return np.rad2deg(lat), np.rad2deg(lon), altitude / 1000.0


@njit(cache=True)
def gcrf_position_to_geodetic(position_gcrf_m, jd_ut1, jd_tt, xp=0.0, yp=0.0):
    """Convert GCRF position [m] to WGS84 geodetic lat/lon [deg], altitude [km]."""
    position_ecef = gcrf_to_ecef_matrix(jd_ut1, jd_tt, xp, yp) @ position_gcrf_m
    return ecef_to_geodetic(position_ecef)
