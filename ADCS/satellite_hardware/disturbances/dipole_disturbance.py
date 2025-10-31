__all__ = ["Dipole_Disturbance"]

import numpy as np
from typing import TYPE_CHECKING
from ADCS.satellite_hardware.disturbances.disturbance import Disturbance
from ADCS.satellite_hardware.actuators.noise import Noise
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.helpers.math_helpers import normalize

class Dipole_Disturbance(Disturbance):
    def __init__(self, dipole_torque: np.ndarray, noise: Noise):
        self.torque_nominal = dipole_torque
        self.noise = noise
        self.current_torque = self.torque_nominal.copy()

    def update(self) -> None:
        self.current_torque = self.torque_nominal + self.noise.get_noise()

    def torque(self, q: np.ndarray, os: Orbital_State) -> np.ndarray:
        vecs = os.get_state_vector(q0=q)

        B_B = vecs["b"]

        return np.cross(self.current_torque, B_B)
    
    def torque_qjac(self, q: np.ndarray, os: Orbital_State) -> np.ndarray:
        vecs = os.get_state_vector(q0=q)

        db_body__dq = vecs["db"]

        return np.cross(self.current_torque, db_body__dq)
    
    def torque_qqhess(self, q: np.ndarray, os: Orbital_State) -> np.ndarray:
        vecs = os.get_state_vector(q0=q)

        ddb_body__dqdq = vecs["ddb"]

        return np.cross(self.current_torque, ddb_body__dqdq)
    
    def torque_valjac(self, q: np.ndarray, os: Orbital_State) -> np.ndarray:
        vecs = os.get_state_vector(q0=q)

        B_B = vecs["b"]

        return np.cross(np.eye(3), B_B)
    
    def torque_qvalhess(self, q: np.ndarray, os: Orbital_State) -> np.ndarray:
        vecs = os.get_state_vector(q0=q)

        db_body__dq = vecs["db"]

        return np.cross(np.expand_dims(np.eye(3),0), np.expand_dims(db_body__dq,1))