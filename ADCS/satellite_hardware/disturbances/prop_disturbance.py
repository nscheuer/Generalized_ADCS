__all__ = ["Prop_Disturbance"]

import numpy as np
from typing import TYPE_CHECKING
from ADCS.satellite_hardware.disturbances.disturbance import Disturbance
from ADCS.satellite_hardware.actuators.noise import Noise
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.helpers.math_helpers import normalize, normed_vec_jac, normed_vec_hess
from ADCS.orbits.universal_constants import EarthConstants

class Prop_Disturbance(Disturbance):
    def __init__(self, torque_nominal: np.ndarray, noise: Noise):
        self.torque_nominal = torque_nominal
        self.noise = noise
        self.current_torque = self.torque_nominal.copy()

    def update(self) -> None:
        self.current_torque = self.torque_nominal + self.noise.get_noise()

    def torque(self):
        return self.current_torque
    
    def torque_qjac(self):
        return np.zeros((3,4))
    
    def torque_qqHess(self):
        return np.zeros((3, 4, 4))
