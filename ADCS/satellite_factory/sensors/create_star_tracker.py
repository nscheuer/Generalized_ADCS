from __future__ import annotations

__all__ = [
    "create_bct_nst",
    "create_terma_t1",
    "create_generic_star_tracker",
]

import numpy as np
from numpy.typing import NDArray

from ADCS.satellite_hardware.sensors.star_tracker import StarTracker
from ADCS.satellite_hardware.actuators import AnisotropicNoise

_ARCSEC2RAD = np.pi / (180.0 * 3600.0)

def create_bct_nst(boresight: NDArray[np.float64] = np.array([0.0, 0.0, 1.0])) -> StarTracker:
    noise = AnisotropicNoise(
        std_cross=6.0 * _ARCSEC2RAD,
        std_roll=40.0 * _ARCSEC2RAD
    )
    return StarTracker(
        boresight=boresight,
        fov=np.deg2rad(10.0),
        anisotropic_noise=noise,
        sun_exclusion=np.deg2rad(45.0)
    )

def create_terma_t1(boresight: NDArray[np.float64] = np.array([0.0, 0.0, 1.0])) -> StarTracker:
    noise = AnisotropicNoise(
        std_cross=2.0 * _ARCSEC2RAD,
        std_roll=15.0 * _ARCSEC2RAD
    )
    return StarTracker(
        boresight=boresight,
        fov=np.deg2rad(22.0),
        anisotropic_noise=noise,
        sun_exclusion=np.deg2rad(30.0)
    )

def create_generic_star_tracker(
    boresight: NDArray[np.float64] = np.array([0.0, 0.0, 1.0]),
    cross_arcsec: float = 10.0,
    roll_arcsec: float = 50.0,
    fov_deg: float = 15.0,
    sun_exclusion_deg: float = 35.0
) -> StarTracker:
    noise = AnisotropicNoise(
        std_cross=cross_arcsec * _ARCSEC2RAD,
        std_roll=roll_arcsec * _ARCSEC2RAD
    )
    return StarTracker(
        boresight=boresight,
        fov=np.deg2rad(fov_deg),
        anisotropic_noise=noise,
        sun_exclusion=np.deg2rad(sun_exclusion_deg)
    )