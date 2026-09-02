__all__ = ['create_pumpkinspace_GPSRM1']

import numpy as np
from typing import Optional, List

from ADCS.satellite_hardware.sensors import GPS 
from ADCS.satellite_hardware.errors import Noise, Bias
from ADCS.helpers.math_helpers import random_n_unit_vec

def create_pumpkinspace_GPSRM1(bias: Bias | None = None, noise: Noise | None = None, estimate_bias: bool = False) -> List[GPS]:
    r"""
    Create a Pumpkin Space GPSRM 1 receiver model.

    Default error model provenance:
        The hardware identity is linked to the Pumpkin Space product listing
        below, but the default bias/noise values in this helper are
        package-level representative simulation values rather than traceable
        vendor or flight-calibration values. Pass explicit ``bias`` and
        ``noise`` objects when a source-backed GPS model is required.

    Source for hardware identity:
        https://www.pumpkinspace.com/store/p58/GNSS_Receiver_Module_%28GPSRM_1%29_Kit.html
    """
    if bias is None:
        e_bias_position = random_n_unit_vec(3)*np.random.uniform(1, 10)
        e_bias_velocity = random_n_unit_vec(3)*np.random.uniform(0.1, 1)
        e_bias = np.stack([e_bias_position, e_bias_velocity])
        std_bias_position = 0.01*np.ones(3)
        std_bias_velocity = 0.0001*np.ones(3)
        std_bias = np.stack([std_bias_position, std_bias_velocity])
        bias = Bias(bias=e_bias, std_bias=std_bias)
    if noise is None:
        e_noise = np.zeros(6)
        std_noise_position = 0.1*np.ones(3)
        std_noise_velocity = 0.1*np.ones(3)
        std_noise = np.stack([std_noise_position, std_noise_velocity])
        noise = Noise(noise=e_noise, std_noise=std_noise)

    return [GPS(bias=bias, noise=noise, estimate_bias=estimate_bias)]

