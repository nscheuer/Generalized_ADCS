import numpy as np
from typing import Optional, List

from ADCS.satellite_hardware.actuators import RW
from ADCS.satellite_hardware.errors import Bias, Noise

def create_cubewheel_smallplus_rw(axis: np.ndarray = np.array([1, 0, 0]), bias: Bias | None = None, noise: Noise | None = None, h_meas_noise: Noise | None = None, estimate_bias: bool = False) -> RW:
    # As used on BeaverCube 2
    if bias is None:
        e_bias = np.random.normal(0.0, 3e-7)
        std_bias = 1e-9
        bias = Bias(bias=e_bias, std_bias=std_bias)
    if noise is None:
        e_noise = 0.0
        std_noise = 1e-6
        noise = Noise(noise=e_noise, std_noise=std_noise)
    if h_meas_noise is None:
        e_h_meas_noise = 0.0
        std_h_meas_noise = 1e-6
        h_meas_noise = Noise(noise=e_h_meas_noise, std_noise=std_h_meas_noise)
        
    max_torque = 0.0023
    J = 5.7e-6
    h = 0.0
    h_max = 0.0036

    return RW(axis=axis, max_torque=max_torque, J=J, h=h, h_max=h_max, bias=bias, noise=noise, h_meas_noise=h_meas_noise, estimate_bias=estimate_bias)


