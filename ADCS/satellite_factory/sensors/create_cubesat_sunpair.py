import numpy as np
from typing import Optional, List

from ADCS.satellite_hardware.sensors import SunPair 
from ADCS.satellite_hardware.actuators import Noise, Bias
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


