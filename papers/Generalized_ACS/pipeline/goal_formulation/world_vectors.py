"""
World vector resolution: resolve named, explicit, or coordinate
targets to ECI unit vectors.

Named vectors reuse the logic from existing Goal subclasses
(Nadir_Goal, Sun_Goal, etc.) but expressed as pure functions.
"""

__all__ = ["resolve_world_vector"]

import numpy as np

from ADCS.pipeline.data import WorldVectorSpec
from ADCS.helpers.math_helpers import normalize
from ADCS.orbits.orbital_state import Orbital_State


def resolve_world_vector(
    u_spec: WorldVectorSpec,
    os: Orbital_State,
) -> np.ndarray:
    """Resolve a world vector specification to an ECI unit vector.

    Parameters
    ----------
    u_spec : WorldVectorSpec
        Target specification (named, explicit vector, or coordinate).
    os : Orbital_State
        Current orbital state providing position, velocity, sun, B-field.

    Returns
    -------
    ndarray, shape (3,)
        Unit vector in ECI frame.
    """
    if u_spec.type == 'named':
        return _resolve_named(u_spec.name, os)
    elif u_spec.type == 'vector':
        return normalize(u_spec.vector)
    elif u_spec.type == 'coordinate':
        return _resolve_coordinate(u_spec.coordinate, os)
    else:
        raise ValueError(f"Unknown u_spec type: {u_spec.type}")


def _resolve_named(name: str, os: Orbital_State) -> np.ndarray:
    """Resolve a named direction to an ECI unit vector.

    Matches the logic in existing Vector_Goal subclasses.
    """
    r = np.asarray(os.R).flatten()
    v = np.asarray(os.V).flatten()

    if name == 'nadir':
        return -normalize(r)
    elif name == 'zenith':
        return normalize(r)
    elif name == 'ram':
        return normalize(v)
    elif name == 'anti_ram':
        return -normalize(v)
    elif name == 'normal':
        return normalize(np.cross(r, v))
    elif name == 'anti_normal':
        return -normalize(np.cross(r, v))
    elif name == 'sun':
        return normalize(np.asarray(os.S).flatten())
    elif name == 'anti_sun':
        return -normalize(np.asarray(os.S).flatten())
    elif name == 'bfield':
        return normalize(np.asarray(os.B).flatten())
    elif name == 'anti_bfield':
        return -normalize(np.asarray(os.B).flatten())
    elif name == 'perp_bfield':
        return _compute_perp_bfield(os)
    else:
        raise ValueError(f"Unknown named direction: {name}")


def _compute_perp_bfield(os: Orbital_State) -> np.ndarray:
    """Compute direction perpendicular to B-field, in the B-V plane.

    Matches PerpBField_Goal logic.
    """
    B_hat = normalize(np.asarray(os.B).flatten())
    V_hat = normalize(np.asarray(os.V).flatten())
    perp = np.cross(B_hat, V_hat)
    if np.linalg.norm(perp) < 1e-6:
        # B and V nearly parallel, use fallback axes
        perp = np.cross(B_hat, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(B_hat, np.array([0.0, 1.0, 0.0]))
    return normalize(perp)


def _resolve_coordinate(coordinate: dict, os: Orbital_State) -> np.ndarray:
    """Resolve a coordinate target to a line-of-sight ECI unit vector.

    Supports {lat, lon, alt} (geodetic) targets. The target position
    is computed in ECEF and rotated to ECI, then the line-of-sight
    direction from the spacecraft is returned.
    """
    if 'lat' in coordinate and 'lon' in coordinate:
        lat = np.radians(coordinate['lat'])
        lon = np.radians(coordinate['lon'])
        alt = coordinate.get('alt', 0.0)  # km above ellipsoid

        # WGS84 ellipsoid constants
        a = 6378.137    # semi-major axis, km
        f = 1 / 298.257223563
        e2 = 2 * f - f**2

        # Geodetic to ECEF
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        x_ecef = (N + alt) * np.cos(lat) * np.cos(lon)
        y_ecef = (N + alt) * np.cos(lat) * np.sin(lon)
        z_ecef = (N * (1 - e2) + alt) * np.sin(lat)
        r_ecef = np.array([x_ecef, y_ecef, z_ecef])

        # ECEF to ECI rotation (Earth rotation)
        # Use the orbital state's GMST for the rotation
        if hasattr(os, 'gmst'):
            theta = os.gmst
        else:
            # Approximate: use Earth rotation rate
            # This is a fallback; the orbital state should provide gmst
            theta = 0.0

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        R_ecef2eci = np.array([
            [cos_t, -sin_t, 0],
            [sin_t,  cos_t, 0],
            [0,      0,     1],
        ])
        r_target_eci = R_ecef2eci @ r_ecef
    elif 'x' in coordinate and 'y' in coordinate and 'z' in coordinate:
        # Direct ECI coordinates
        r_target_eci = np.array([
            coordinate['x'], coordinate['y'], coordinate['z']
        ])
    else:
        raise ValueError(f"Unrecognized coordinate format: {coordinate}")

    # Line-of-sight direction from spacecraft
    r_sat = np.asarray(os.R).flatten()
    direction = r_target_eci - r_sat
    return normalize(direction)
