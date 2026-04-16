from __future__ import annotations

__all__ = [
    "create_generic_earth_horizon",
    "create_irst_horizon_sensor",
]

import numpy as np
from numpy.typing import NDArray

from ADCS.satellite_hardware.sensors.earth_horizon import EarthHorizonSensor
from ADCS.satellite_hardware.errors import Noise


def create_generic_earth_horizon(
    boresight: NDArray[np.float64] = np.array([0.0, 0.0, -1.0]),
    fov_deg: float = 90.0,
    noise_deg: float = 0.5,
) -> EarthHorizonSensor:
    r"""
    Create a generic Earth horizon sensor with configurable parameters.

    :param boresight: Sensor boresight in the body frame. Default nadir-pointing.
    :type boresight: numpy.ndarray
    :param fov_deg: Half-cone field of view [deg].
    :type fov_deg: float
    :param noise_deg: 1-sigma measurement noise per axis [deg].
    :type noise_deg: float
    :return: Configured Earth horizon sensor.
    :rtype: :class:`~ADCS.satellite_hardware.sensors.earth_horizon.EarthHorizonSensor`
    """
    noise_rad = np.deg2rad(noise_deg)
    noise = Noise(
        noise=np.zeros(3),
        std_noise=np.array([noise_rad, noise_rad, noise_rad]),
    )
    return EarthHorizonSensor(
        boresight=boresight,
        fov=np.deg2rad(fov_deg),
        noise=noise,
    )


def create_irst_horizon_sensor(
    boresight: NDArray[np.float64] = np.array([0.0, 0.0, -1.0]),
) -> EarthHorizonSensor:
    r"""
    Create an infrared scanning telescope (IRST) Earth horizon sensor.

    Typical performance for a miniaturized IR horizon sensor:

    * Half-cone FOV: 60 deg
    * Nadir accuracy: ~0.25 deg (1-sigma per axis)

    :param boresight: Sensor boresight in the body frame.
    :type boresight: numpy.ndarray
    :return: Configured IRST horizon sensor.
    :rtype: :class:`~ADCS.satellite_hardware.sensors.earth_horizon.EarthHorizonSensor`
    """
    noise_rad = np.deg2rad(0.25)
    noise = Noise(
        noise=np.zeros(3),
        std_noise=np.array([noise_rad, noise_rad, noise_rad]),
    )
    return EarthHorizonSensor(
        boresight=boresight,
        fov=np.deg2rad(60.0),
        noise=noise,
    )
