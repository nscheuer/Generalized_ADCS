import numpy as np
from typing import Optional, List

from ADCS.satellite_hardware.actuators import MTQ
from ADCS.satellite_hardware.errors import Noise, Bias
from ADCS.helpers.math_helpers import random_n_unit_vec

def create_isis_magnetorquer_board(axes: np.ndarray = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), bias: List[Bias] | None = None, noise: List[Noise] | None = None, estimate_bias: bool = False) -> List[MTQ]:
    # https://satsearch.co/products/isis-isis-magnetorquer-board-i-mtq
    if bias is None:
        e_bias = random_n_unit_vec(3)*np.random.uniform(0.01, 0.15)
        e_bias = np.zeros(3)
        std_bias = 0.000001*np.ones(3)
        std_bias = np.zeros(3)
        bias = [Bias(bias=e_bias[j], std_bias=std_bias[j]) for j in range(3)]
    if noise is None:
        e_noise = np.zeros(3)
        std_noise = 0.0001*np.ones(3)
        std_noise = np.zeros(3)
        noise = [Noise(noise=e_noise[j], std_noise=std_noise[j]) for j in range(3)]
    mtq_max = 0.2

    return [MTQ(axis=axes[j], max_torque=mtq_max, bias=bias[j], noise=noise[j], estimate_bias=estimate_bias) for j in range(3)]


