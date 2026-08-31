__all__ = [
    'create_ICM20948_IMU',
    'create_adis16405_gyros',
    'create_bmx055_gyros',
    'create_itg3200_gyros',
]

import numpy as np
from typing import Optional, List

from ADCS.satellite_hardware.sensors import Gyro 
from ADCS.satellite_hardware.errors import Noise, Bias
from ADCS.helpers.math_helpers import random_n_unit_vec

def create_ICM20948_IMU(axes: np.ndarray = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), bias: Bias | None = None, noise: Noise | None = None, estimate_bias: bool = False) -> List[Gyro]:
    r"""
    Create an ICM-20948 gyro triad.

    Default error model provenance:
        The default bias/noise values in this helper are package-level
        representative simulation values for BeaverCube-style examples. They
        are not currently tied to a published ICM-20948 paper, data sheet table,
        or BeaverCube flight-calibration result. Pass explicit ``bias`` and
        ``noise`` objects when a traceable mission or vendor model is required.
    """
    if bias is None:
        e_bias = random_n_unit_vec(3)*np.random.uniform(0.01,0.2)*(np.pi/180.0)
        std_bias = 0.0004*np.pi/180.0*np.ones(3)
        bias = [Bias(bias=e_bias[j], std_bias=std_bias[j]) for j in range(3)]
    if noise is None:
        e_noise = np.zeros(3)
        std_noise = 0.03*np.pi/180.0*np.ones(3)
        noise = [Noise(noise=e_noise[j], std_noise=std_noise[j]) for j in range(3)]

    return [Gyro(axis=axes[j], bias=bias[j], noise=noise[j], estimate_bias=estimate_bias) for j in range(3)]


def create_adis16405_gyros(
    axes: np.ndarray = np.eye(3),
    bias: List[Bias] | None = None,
    noise: List[Noise] | None = None,
    estimate_bias: bool = False,
) -> List[Gyro]:
    r"""
    Create the Analog Devices ADIS16405 gyro channels used on RAX-1 and RAX-2.

    RAX used one ADIS16405 IMU on the attitude determination board.

    Default error model provenance:
        ``bias`` defaults to a Gaussian initial bias with 3 deg/s one-sigma
        component error, and ``std_bias`` uses the 0.007 deg/s in-run bias
        stability value. ``noise`` defaults to 0.9 deg/s output-noise RMS.
        These values are ADIS16405 component characterization values reported
        for the RAX attitude-determination system; they are not a single fixed
        RAX flight-estimated bias vector.

        The RAX estimator simulation also reported angle-random-walk
        ``4.89e-4 rad/s^(1/2)`` and rate-random-walk
        ``3.14e-5 rad/s^(3/2)`` parameters, but those are not represented
        directly by this scalar white-noise
        :class:`~ADCS.satellite_hardware.sensors.Gyro` model.

    Source:
        J. C. Springmann, *Satellite Attitude Determination with Low-Cost
        Sensors*, Ph.D. dissertation, University of Michigan, 2013.
        https://deepblue.lib.umich.edu/bitstream/handle/2027.42/102312/jspringm_1.pdf?sequence=1
    """
    n_axes = len(axes)
    if bias is None:
        bias_std = 3.0 * np.pi / 180.0
        e_bias = np.random.normal(0.0, bias_std, size=n_axes)
        bias = [Bias(bias=e_bias[j], std_bias=0.007 * np.pi / 180.0) for j in range(n_axes)]
    if noise is None:
        std_noise = 0.9 * np.pi / 180.0
        noise = [Noise(noise=0.0, std_noise=std_noise) for _ in range(n_axes)]

    return [Gyro(axis=axes[j], bias=bias[j], noise=noise[j], estimate_bias=estimate_bias) for j in range(n_axes)]


def create_bmx055_gyros(
    axes: np.ndarray = np.eye(3),
    bias: List[Bias] | None = None,
    noise: List[Noise] | None = None,
    estimate_bias: bool = False,
) -> List[Gyro]:
    r"""
    Create one Bosch Sensortec BMX055 gyroscope triad used on MOVE-II.

    The BMX055 gyroscope is represented as three one-dimensional
    :class:`~ADCS.satellite_hardware.sensors.Gyro` channels.

    Default error model provenance:
        ``bias`` defaults to the MOVE-II HIL simulation vector
        ``[1.75e-3, 3.49e-3, -1.75e-3] rad/s`` for one BMX055 triad. ``noise``
        defaults to additive Gaussian white noise with ``sigma = 8.727e-4
        rad/s``. The source also models scale factor,
        misalignment, nonorthogonality, quantization, time sampling,
        low-pass filtering, and gyro-bias random walk; those terms are not
        represented directly by this scalar gyro factory.

    Source:
        `Hardware-in-the-Loop Verification of the Distributed,
        Magnetorquer-Based ADCS of the CubeSat MOVE-II
        <https://mediatum.ub.tum.de/doc/1483411/document.pdf>`__
    """
    n_axes = len(axes)
    if bias is None:
        triad_bias = np.array([1.75e-3, 3.49e-3, -1.75e-3])
        bias = [Bias(bias=triad_bias[j], std_bias=0.0) for j in range(n_axes)]
    if noise is None:
        noise = [Noise(noise=0.0, std_noise=8.727e-4) for _ in range(n_axes)]

    return [Gyro(axis=axes[j], bias=bias[j], noise=noise[j], estimate_bias=estimate_bias) for j in range(n_axes)]


def create_itg3200_gyros(
    axes: np.ndarray = np.eye(3),
    bias: List[Bias] | None = None,
    noise: List[Noise] | None = None,
    estimate_bias: bool = False,
) -> List[Gyro]:
    r"""
    Create one ESTCube-1 InvenSense ITG-3200 gyro triad.

    The ITG-3200 gyroscope is represented as three one-dimensional
    :class:`~ADCS.satellite_hardware.sensors.Gyro` channels.

    Default error model provenance:
        ``bias`` defaults to a uniform initial bias over +/-0.5 deg/s per
        measurement channel. ``noise`` defaults to Gaussian 1.8 deg/s standard
        deviation. These are the ESTCube-1 preflight simulator values. The
        ground characterization also reported 0.9 deg/s statistical noise, and
        flight calibration reported approximately 0.5 deg/s expanded
        uncertainty; those are documented here but are not the default.

    Source:
        J. Slavinskis et al., "ESTCube-1 attitude determination system flight
        results," *Journal of Aerospace Engineering*, 2016.
        https://ascelibrary.org/doi/10.1061/%28ASCE%29AS.1943-5525.0000504
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
