import numpy as np
from typing import Optional, List

from ADCS.satellite_hardware.actuators import RW, Noise, Bias

def create_cubewheel_smallplus_rw(axis: np.ndarray = np.array([1, 0, 0]), bias: Bias | None = None, noise: Noise | None = None, estimate_bias: bool = False) -> RW:
    # As used on BeaverCube 2
    if bias is None:
        e_bias = np.random.normal(0.0, 3e-7)
        std_bias = 1e-9
        bias = Bias(bias=e_bias, std_bias=std_bias)
    if noise is None:
        e_noise = 0.0
        std_noise = 1e-6
        noise = Noise(noise=e_noise, std_noise=std_noise)
    mtq_max = 0.002

    return RW(axis=axis, max_torque=mtq_max, bias=bias, noise=noise, estimate_bias=estimate_bias)


