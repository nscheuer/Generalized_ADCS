__all__ = ['create_Clydespace_3U_array', 'create_hamamatsu_s3931_sun_sensors']

import numpy as np
from typing import Optional, List

from ADCS.satellite_hardware.sensors import SunPair, SunSensor
from ADCS.satellite_hardware.errors import Noise, Bias
from ADCS.helpers.math_helpers import random_n_unit_vec

def create_Clydespace_3U_array(axis: np.ndarray = np.array([1, 0, 0]), bias: Bias | None = None, noise: Noise | None = None, estimate_bias: bool = False) -> List[SunPair]:
    # As used on BeaverCube 1 & 2
    sun_eff = 0.3
    if bias is None:
        e_bias = np.random.uniform(0,0.2)*sun_eff
        std_bias = 0.00001*sun_eff
        bias = Bias(bias=e_bias, std_bias=std_bias) 
    if noise is None:
        e_noise = 0
        std_noise = 0.001*sun_eff
        noise = Noise(noise=e_noise, std_noise=std_noise)

    return [SunPair(axis=axis, efficiency=(sun_eff, sun_eff), bias=bias, noise=noise, estimate_bias=estimate_bias)]


def create_hamamatsu_s3931_sun_sensors(
    axes: np.ndarray = np.repeat(np.vstack((np.eye(3), -np.eye(3))), 2, axis=0),
    bias: List[Bias] | None = None,
    noise: List[Noise] | None = None,
    estimate_bias: bool = False,
) -> List[SunSensor]:
    r"""
    Create the ESTCube-1 Hamamatsu S3931 one-dimensional Sun sensor channels.

    ESTCube-1 had six custom Sun sensor assemblies, one per spacecraft face.
    Each assembly used two Hamamatsu S3931 one-dimensional PSDs under
    perpendicular slits, so this factory returns twelve one-dimensional
    :class:`~ADCS.satellite_hardware.sensors.SunSensor` channels. The preflight
    simulator used angular noise of approximately 1.25 deg and initial angular
    bias uniformly distributed over +/-1 deg. The useful flight region was
    approximately +/-36.7 deg, with incidence-angle-dependent flight covariance;
    the current sensor model does not encode that finite field-of-view or
    covariance schedule, so these defaults approximate angular errors as scalar
    cosine-measurement perturbations.

    Source: `ESTCube-1 attitude determination flight results
    <https://ascelibrary.org/doi/10.1061/%28ASCE%29AS.1943-5525.0000504>`__
    """
    n_axes = len(axes)
    if bias is None:
        bias_bound = np.sin(1.0 * np.pi / 180.0)
        e_bias = np.random.uniform(-bias_bound, bias_bound, size=n_axes)
        bias = [Bias(bias=e_bias[j], std_bias=0.0, bounds=(-bias_bound, bias_bound)) for j in range(n_axes)]
    if noise is None:
        std_noise = np.sin(1.25 * np.pi / 180.0)
        noise = [Noise(noise=0.0, std_noise=std_noise) for _ in range(n_axes)]

    return [
        SunSensor(axis=axes[j], efficiency=1.0, bias=bias[j], noise=noise[j], estimate_bias=estimate_bias)
        for j in range(n_axes)
    ]
