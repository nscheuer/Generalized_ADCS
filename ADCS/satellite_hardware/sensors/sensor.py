__all__ = ["Sensor"]

import numpy as np

from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.actuators import Noise, Bias

class Sensor:
    def __init__(self, sample_time: float = 0.1, output_length: int = 1, bias: Bias = None, noise: Noise = None, estimate_bias: bool = False):
        if bias:
            self.bias = bias
        else:
            self.bias = Bias(bias=np.zeros(6), std_bias=np.zeros(6))
        if noise:
            self.noise = noise
        else:
            self.noise = Noise(noise=np.zeros(6), std_noise=np.zeros(6))
        self.sample_time = sample_time
        self.output_length = output_length

    def reading(self, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        reading = self.clean_reading(x=x, os=os)
        if self.bias:
            reading += self.bias.get_bias(os.J2000)

        if self.noise:
            reading += self.noise.get_noise()

        return reading
    
    def basestate_jac(self, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        return np.zeros((7, self.output_length))
    
    def bias_jac(self, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        if self.bias:
            return np.eye(self.output_length)
        else:
            return np.zeros((0, self.output_length))