"""
Factory functions for creating star tracker sensors.

Provides convenience functions for common star tracker models with
realistic specifications from manufacturer datasheets.

References:
    [1] BCT Nano Star Tracker Datasheet
    [2] Terma T1 Star Tracker Datasheet
    [3] Liebe, C.C., "Star Trackers for Attitude Determination",
        IEEE AES Magazine (1995, 2002)
"""
from __future__ import annotations

__all__ = [
    "create_bct_nst",
    "create_terma_t1",
    "create_generic_star_tracker",
]

import numpy as np
from numpy.typing import NDArray

from ADCS.satellite_hardware.sensors.star_tracker import StarTracker


# Conversion factor: arcseconds to radians
_ARCSEC2RAD = np.pi / (180.0 * 3600.0)


def create_bct_nst(
    boresight: NDArray[np.float64] = np.array([0.0, 0.0, 1.0])
) -> StarTracker:
    """Create Blue Canyon Technologies Nano Star Tracker (NST).

    The BCT NST is a compact, high-performance star tracker designed
    for small satellites and CubeSats.

    Specifications:
        - Cross-boresight accuracy: 6 arcsec (1σ)
        - Roll accuracy: 40 arcsec (1σ)
        - FOV: 10° x 12° (uses 10° as conservative estimate)
        - Sun exclusion: 45°
        - Mass: ~350g
        - Power: ~1W

    Args:
        boresight: Boresight direction in body frame.
            Default: +Z axis [0, 0, 1].

    Returns:
        Configured StarTracker instance.

    Reference:
        BCT Nano Star Tracker Datasheet

    Example:
        >>> tracker = create_bct_nst()
        >>> # Mount on -Z face looking outward
        >>> tracker = create_bct_nst(boresight=np.array([0, 0, -1]))
    """
    return StarTracker(
        boresight=boresight,
        fov=np.deg2rad(10.0),
        cross_noise_std=6.0 * _ARCSEC2RAD,
        roll_noise_std=40.0 * _ARCSEC2RAD,
        sun_exclusion=np.deg2rad(45.0)
    )


def create_terma_t1(
    boresight: NDArray[np.float64] = np.array([0.0, 0.0, 1.0])
) -> StarTracker:
    """Create Terma T1 Star Tracker.

    The Terma T1 is a high-accuracy star tracker used on many
    scientific and commercial missions.

    Specifications:
        - Cross-boresight accuracy: 2 arcsec (1σ)
        - Roll accuracy: 15 arcsec (1σ)
        - FOV: 22° x 22°
        - Sun exclusion: 30°
        - Mass: ~2.5 kg
        - Power: ~7W

    Args:
        boresight: Boresight direction in body frame.
            Default: +Z axis [0, 0, 1].

    Returns:
        Configured StarTracker instance.

    Reference:
        Terma T1 Star Tracker Datasheet

    Example:
        >>> tracker = create_terma_t1()
    """
    return StarTracker(
        boresight=boresight,
        fov=np.deg2rad(22.0),
        cross_noise_std=2.0 * _ARCSEC2RAD,
        roll_noise_std=15.0 * _ARCSEC2RAD,
        sun_exclusion=np.deg2rad(30.0)
    )


def create_generic_star_tracker(
    boresight: NDArray[np.float64] = np.array([0.0, 0.0, 1.0]),
    cross_arcsec: float = 10.0,
    roll_arcsec: float = 50.0,
    fov_deg: float = 15.0,
    sun_exclusion_deg: float = 35.0
) -> StarTracker:
    """Create a generic star tracker with custom parameters.

    Use this function when you need to specify custom noise and
    field of view parameters, or to model a specific star tracker
    not included in the preset functions.

    Args:
        boresight: Boresight direction in body frame.
            Default: +Z axis [0, 0, 1].
        cross_arcsec: Cross-boresight noise standard deviation in arcseconds.
            Typical range: 1-30 arcsec. Default: 10 arcsec.
        roll_arcsec: Roll noise standard deviation in arcseconds.
            Typical range: 5-100 arcsec. Default: 50 arcsec.
        fov_deg: Field of view in degrees (full angle).
            Typical range: 8-25 degrees. Default: 15 degrees.
        sun_exclusion_deg: Sun exclusion angle in degrees.
            Tracker is blinded when sun is closer than this.
            Typical range: 25-55 degrees. Default: 35 degrees.

    Returns:
        Configured StarTracker instance.

    Reference:
        Liebe (2002), Section III for typical noise values
        Liebe (2002), Section IV-A for typical sun exclusion angles

    Example:
        >>> # High-accuracy tracker for a science mission
        >>> tracker = create_generic_star_tracker(
        ...     cross_arcsec=3.0,
        ...     roll_arcsec=20.0,
        ...     fov_deg=20.0,
        ...     sun_exclusion_deg=30.0
        ... )
    """
    return StarTracker(
        boresight=boresight,
        fov=np.deg2rad(fov_deg),
        cross_noise_std=cross_arcsec * _ARCSEC2RAD,
        roll_noise_std=roll_arcsec * _ARCSEC2RAD,
        sun_exclusion=np.deg2rad(sun_exclusion_deg)
    )
