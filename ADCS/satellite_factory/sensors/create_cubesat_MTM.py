__all__ = ['create_isis_magnetometer', 'create_hmc5883l_magnetometers']

import numpy as np
from typing import Optional, List

from ADCS.satellite_hardware.sensors import MTM 
from ADCS.satellite_hardware.errors import Noise, Bias
from ADCS.helpers.math_helpers import random_n_unit_vec

def create_isis_magnetometer(axes: np.ndarray = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), bias: Bias | None = None, noise: Noise | None = None, estimate_bias: bool = False) -> List[MTM]:
    # https://satsearch.co/products/isis-isis-magnetorquer-board-i-mtq
    if bias is None:
        e_bias = random_n_unit_vec(3)*np.random.uniform(1e-9,1e-7)
        std_bias = 1e-9*np.ones(3)
        bias = [Bias(bias=e_bias[j], std_bias=std_bias[j]) for j in range(3)]
    if noise is None:
        e_noise = np.zeros(3)
        std_noise = 3*1e-7*np.ones(3)
        noise = [Noise(noise=e_noise[j], std_noise=std_noise[j]) for j in range(3)]

    return [MTM(axis=axes[j], bias=bias[j], noise=noise[j], estimate_bias=estimate_bias) for j in range(3)]


def create_hmc5883l_magnetometers(
    axes: np.ndarray = np.vstack((np.eye(3), np.eye(3))),
    bias: List[Bias] | None = None,
    noise: List[Noise] | None = None,
    estimate_bias: bool = False,
) -> List[MTM]:
    r"""
    Create the ESTCube-1 Honeywell HMC5883L magnetometer set.

    ESTCube-1 used two three-axis HMC5883L magnetometers. The preflight
    simulator modeled magnetic-field direction noise of approximately 1.6 deg
    and initial magnetometer bias of +/-2400 nT. Since this package's
    :class:`~ADCS.satellite_hardware.sensors.MTM` model uses scalar magnetic
    field measurements, the default noise approximates 1.6 deg direction error
    as a transverse component error for a representative 50 microtesla field.

    Source: `ESTCube-1 attitude determination flight results
    <https://ascelibrary.org/doi/10.1061/%28ASCE%29AS.1943-5525.0000504>`__
    """
    n_axes = len(axes)
    if bias is None:
        bias_bound = 2400e-9
        e_bias = np.random.uniform(-bias_bound, bias_bound, size=n_axes)
        bias = [Bias(bias=e_bias[j], std_bias=0.0, bounds=(-bias_bound, bias_bound)) for j in range(n_axes)]
    if noise is None:
        representative_field = 50e-6
        std_noise = representative_field * np.sin(1.6 * np.pi / 180.0)
        noise = [Noise(noise=0.0, std_noise=std_noise) for _ in range(n_axes)]

    return [MTM(axis=axes[j], bias=bias[j], noise=noise[j], estimate_bias=estimate_bias) for j in range(n_axes)]
