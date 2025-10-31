__all__ = ["Actuator"]

import numpy as np
from ADCS.satellite_hardware.actuators.bias import Bias
from ADCS.satellite_hardware.actuators.noise import Noise
from ADCS.orbits.universal_constants import TimeConstants

class Actuator:
    def __init__(self, axis: np.ndarray, u_max: float, bias: Bias = None, noise: Noise = None, estimate_bias: bool = False) -> None:
        self.axis = axis
        self.u_max = u_max
        if bias:
            self.bias: Bias = bias
        else:
            self.bias = Bias()
        if noise:
            self.noise: Noise = noise
        else:
            self.noise = Noise()
        self.estimate_bias: bool = estimate_bias
        self.last_bias_time: float = float('nan')

    def torque(self, command: float, j2000: float, bias: bool = False, noise: bool = False) -> float:
        return np.ndarray([0, 0, 0])
    
    def storage_torque(self, command: float, j2000: float, bias: bool = False, noise: bool = False) -> float:
        return 0

    
