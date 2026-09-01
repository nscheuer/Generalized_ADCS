from __future__ import annotations

__all__ = [
    "create_aeroastro_mst",
    "create_bct_nst",
    "create_terma_t1",
    "create_generic_star_tracker",
    "create_bct_nst_quaternion",
    "create_generic_star_tracker_quaternion",
]

import numpy as np
from numpy.typing import NDArray

from ADCS.satellite_hardware.sensors.star_tracker import StarTracker
from ADCS.satellite_hardware.sensors.star_tracker_quaternion import StarTrackerQuaternion
from ADCS.satellite_hardware.errors import AnisotropicNoise, Bias, Noise

_ARCSEC2RAD = np.pi / (180.0 * 3600.0)


def create_aeroastro_mst(
    boresight: NDArray[np.float64] = np.array([0.0, 0.0, 1.0]),
    bias: Bias | None = None,
    anisotropic_noise: AnisotropicNoise | None = None,
    estimate_bias: bool = False,
) -> StarTracker:
    r"""
    Create the AeroAstro Miniature Star Tracker used on BRITE-Austria.

    The source gives mass, dimensions, update rate, and attitude accuracy
    specifications. No source-backed Gaussian noise or fixed bias model is
    assigned here, so the default sensor error model is ideal.
    """
    return StarTracker(
        boresight=boresight,
        bias=bias,
        anisotropic_noise=anisotropic_noise,
        estimate_bias=estimate_bias,
        fov=np.deg2rad(10.0),
        sun_exclusion=np.deg2rad(30.0),
    )

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


def create_bct_nst_quaternion(
    boresight: NDArray[np.float64] = np.array([0.0, 0.0, 1.0]),
) -> StarTrackerQuaternion:
    r"""
    Create a BCT Nano Star Tracker in quaternion output mode.

    Uses the same noise specification as the BCT NST vector tracker
    (6 arcsec cross-boresight, 40 arcsec roll) mapped to a 4-element
    quaternion noise model, with a wider FOV to ensure multiple stars
    are visible.

    :param boresight: Sensor boresight in the body frame.
    :type boresight: numpy.ndarray
    :return: Quaternion star tracker with BCT NST noise parameters.
    :rtype: :class:`~ADCS.satellite_hardware.sensors.star_tracker_quaternion.StarTrackerQuaternion`
    """
    cross_rad = 6.0 * _ARCSEC2RAD
    noise = Noise(
        noise=np.zeros(4),
        std_noise=np.array([cross_rad, cross_rad, cross_rad, cross_rad]),
    )
    return StarTrackerQuaternion(
        boresight=boresight,
        fov=np.deg2rad(20.0),
        noise=noise,
        sun_exclusion=np.deg2rad(45.0),
    )


def create_generic_star_tracker_quaternion(
    boresight: NDArray[np.float64] = np.array([0.0, 0.0, 1.0]),
    noise_arcsec: float = 10.0,
    fov_deg: float = 20.0,
    sun_exclusion_deg: float = 35.0,
    min_stars: int = 2,
) -> StarTrackerQuaternion:
    r"""
    Create a configurable quaternion-output star tracker.

    :param boresight: Sensor boresight in the body frame.
    :type boresight: numpy.ndarray
    :param noise_arcsec: Per-component quaternion noise [arcsec].
    :type noise_arcsec: float
    :param fov_deg: Full-angle field of view [deg].
    :type fov_deg: float
    :param sun_exclusion_deg: Sun exclusion angle [deg].
    :type sun_exclusion_deg: float
    :param min_stars: Minimum number of stars for a valid solution.
    :type min_stars: int
    :return: Configured quaternion star tracker.
    :rtype: :class:`~ADCS.satellite_hardware.sensors.star_tracker_quaternion.StarTrackerQuaternion`
    """
    noise_rad = noise_arcsec * _ARCSEC2RAD
    noise = Noise(
        noise=np.zeros(4),
        std_noise=np.array([noise_rad] * 4),
    )
    return StarTrackerQuaternion(
        boresight=boresight,
        fov=np.deg2rad(fov_deg),
        noise=noise,
        sun_exclusion=np.deg2rad(sun_exclusion_deg),
        min_stars=min_stars,
    )
