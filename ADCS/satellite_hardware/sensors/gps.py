__all__ = ["GPS"]

from .sensor import Sensor

import numpy as np
from scipy.linalg import block_diag

from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.actuators import Noise, Bias
from ADCS.helpers.math_constants import MathConstants

class GPS(Sensor):
    def __init__(self, sample_time: float = 0.1, bias: Bias = None, noise: Noise = None, estimate_bias: bool = False):
        self.attitude_sensor = False
        super().__init__(sample_time=sample_time, output_length=6, bias=bias, noise=noise, estimate_bias=estimate_bias)

    def clean_reading(self, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        ecef = os.ECEF
        v = os.V
        return np.concatenate([ecef, os.eci_to_ecef(v)])
    
    def bias_jac(self, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        if self.bias:
            return np.eye(6)
        else:
            return np.zeros((0, 6))
        
    def orbitRV_jac(self, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        mat = np.stack([os.eci_to_ecef(j) for j in MathConstants.unitvecs]).T
        return block_diag(mat, mat)

