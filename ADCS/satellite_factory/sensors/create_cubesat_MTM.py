import numpy as np
from typing import Optional, List

from ADCS.satellite_hardware.sensors import MTM 
from ADCS.satellite_hardware.actuators import Noise, Bias
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


