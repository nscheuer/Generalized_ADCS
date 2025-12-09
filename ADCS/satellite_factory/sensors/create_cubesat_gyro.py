import numpy as np
from typing import Optional, List

from ADCS.satellite_hardware.sensors import Gyro 
from ADCS.satellite_hardware.actuators import Noise, Bias
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


