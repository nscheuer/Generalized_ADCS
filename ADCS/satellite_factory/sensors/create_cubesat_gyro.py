__all__ = ['create_ICM20948_IMU', 'create_itg3200_gyros']

import numpy as np
from typing import Optional, List

from ADCS.satellite_hardware.sensors import Gyro 
from ADCS.satellite_hardware.errors import Noise, Bias
from ADCS.helpers.math_helpers import random_n_unit_vec

def create_ICM20948_IMU(axes: np.ndarray = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), bias: Bias | None = None, noise: Noise | None = None, estimate_bias: bool = False) -> List[Gyro]:
    # As used on BeaverCube 1 & 2
    if bias is None:
        e_bias = random_n_unit_vec(3)*np.random.uniform(0.01,0.2)*(np.pi/180.0)
        std_bias = 0.0004*np.pi/180.0*np.ones(3)
        bias = [Bias(bias=e_bias[j], std_bias=std_bias[j]) for j in range(3)]
    if noise is None:
        e_noise = np.zeros(3)
        std_noise = 0.03*np.pi/180.0*np.ones(3)
        noise = [Noise(noise=e_noise[j], std_noise=std_noise[j]) for j in range(3)]

    return [Gyro(axis=axes[j], bias=bias[j], noise=noise[j], estimate_bias=estimate_bias) for j in range(3)]


def create_itg3200_gyros(
    axes: np.ndarray = np.tile(np.eye(3), (4, 1)),
    bias: List[Bias] | None = None,
    noise: List[Noise] | None = None,
    estimate_bias: bool = False,
) -> List[Gyro]:
    r"""
    Create the ESTCube-1 InvenSense ITG-3200 gyro set.

    ESTCube-1 carried four ITG-3200 three-axis gyros on the ADCS sensor board,
    represented here as twelve one-dimensional
    :class:`~ADCS.satellite_hardware.sensors.Gyro` channels. The preflight
    simulator used gyro noise of approximately 1.8 deg/s and initial bias
    uniformly distributed over +/-0.5 deg/s per measurement channel. Flight
    calibration reported approximately 0.5 deg/s expanded uncertainty.

    Source: `ESTCube-1 attitude determination flight results
    <https://ascelibrary.org/doi/10.1061/%28ASCE%29AS.1943-5525.0000504>`__
    """
    n_axes = len(axes)
    if bias is None:
        bias_bound = 0.5 * np.pi / 180.0
        e_bias = np.random.uniform(-bias_bound, bias_bound, size=n_axes)
        bias = [Bias(bias=e_bias[j], std_bias=0.0, bounds=(-bias_bound, bias_bound)) for j in range(n_axes)]
    if noise is None:
        std_noise = 1.8 * np.pi / 180.0
        noise = [Noise(noise=0.0, std_noise=std_noise) for _ in range(n_axes)]

    return [Gyro(axis=axes[j], bias=bias[j], noise=noise[j], estimate_bias=estimate_bias) for j in range(n_axes)]
