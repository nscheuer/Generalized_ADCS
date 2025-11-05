__all__ = ["Actuator"]

import numpy as np
from ADCS.satellite_hardware.actuators.bias import Bias
from ADCS.satellite_hardware.actuators.noise import Noise
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.helpers.math_helpers import normalize

class Actuator:
    def __init__(self, axis: np.ndarray, u_max: float, bias: Bias = None, noise: Noise = None, estimate_bias: bool = False) -> None:
        self.axis = normalize(axis)
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

    def torque(self, command: float, x: np.ndarray, os: Orbital_State, bias: bool = False, noise: bool = False) -> float:
        return np.ndarray([0, 0, 0])
    
    def storage_torque(self, command: float, j2000: float, bias: bool = False, noise: bool = False) -> float:
        return 0
    
    def dtorq__du(self, command: float, q: np.ndarray, os: Orbital_State) -> np.ndarray:
        return np.zeros((1, 3))
    
    def dtorq__dbias(self, command: float, q: np.ndarray, os: Orbital_State) -> np.ndarray:
        if self.bias:
            return self.dtorq__du(command=command, q=q, os=os)
        else:
            return np.zeros((0, 3))
        
    def dtorq__dbasestate(self, command: float, q: np.ndarray, os: Orbital_State) -> np.ndarray:
        return np.zeros((7, 3))
    
    def dtorq__dh(self, command: float, q: np.ndarray, os: Orbital_State) -> np.ndarray:
        return np.zeros((0,3))
    
    def ddtorq__dudu(self, command: float, q: np.ndarray, os: Orbital_State) -> np.ndarray:
        return np.zeros((1, 1, 3))
    
    def ddtorq__dudbias(self, command: float, q: np.ndarray, os: Orbital_State) -> np.ndarray:
        if self.bias:
            return self.ddtorq__dudu(command=command, q=q, os=os)
        else:
            return np.zeros((1, 0, 3))
        
    def ddtorq__dudbasestate(self, command: float, q: np.ndarray, os: Orbital_State) -> np.ndarray:
        return np.zeros((1, 7, 3))
    
    def ddtorq__dudh(self, command: float, q: np.ndarray, os: Orbital_State) -> np.ndarray:
        return np.zeros((1, 0, 3))
    
    def ddtorq__dbiasdbias(self, command: float, q: np.ndarray, os: Orbital_State) -> np.ndarray:
        if self.bias:
            return self.ddtorq__dudu(command=command, q=q, os=os)
        else:
            return np.zeros((0, 0, 3))
        
    def ddtorq_dbiasdbasestate(self, command: float, q: np.ndarray, os: Orbital_State) -> np.ndarray:
        if self.bias:
            return self.ddtorq__dudbasestate(command=command, q=q, os=os)
        else:
            return np.zeros((0, 7, 3))
        
    def ddtotq__dbiasdh(self, command: float, q: np.ndarray, os: Orbital_State) -> np.ndarray:
        if self.bias:
            return self.ddtorq__dudh(command=command, q=q, os=os)
        else:
            return np.zeros((0, 0, 3))
        
    def ddtorq__dbasestatedh(self, command: float, q: np.ndarray, os: Orbital_State) -> np.ndarray:
        return np.zeros((7, 0, 3))
    
    def ddtorq__dhdh(self, command: float, q: np.ndarray, os: Orbital_State) -> np.ndarray:
        return np.zeros((0, 0, 3))
    
    def dstor_torq__du(self, command: float, q: np.ndarray, os: Orbital_State) -> np.ndarray:
        return np.zeros(1, 0)
    
    def dstor_torq__dbias(self, command: float, q: np.ndarray, os: Orbital_State) -> np.ndarray:
        if self.bias:
            return self.dstor_torq__du(command=command, q=q, os=os)
        else:
            return np.zeros((0, 0))
        
    def dstor_torq__dbasestate(self, command: float, q: np.ndarray, os: Orbital_State) -> np.ndarray:
        return np.zeros((7, 0))
        
    def dstor_torq__dh(self, command: float, q: np.ndarray, os: Orbital_State) -> np.ndarray:
        return np.zeros((0, 0))
    
    def ddstor_torq__dudu(self, command: float, q: np.ndarray, os: Orbital_State) -> np.ndarray:
        return np.zeros((1, 1, 0))
    
    def ddstor_torq__dudbias(self, command: float, q: np.ndarray, os: Orbital_State) -> np.ndarray:
        if self.bias:
            return self.ddstor_torq__dudu(command=command, q=q, os=os)
        else:
            return np.zeros((1, 0, 0))

    def ddstor_torq__dudbasestate(self, command: float, q: np.ndarray, os: Orbital_State) -> np.ndarray:
        return np.zeros((1, 7, 0))
    
    def ddstor_torq__dudh(self, command: float, q: np.ndarray, os: Orbital_State) -> np.ndarray:
        return np.zeros((1, 0, 0))
    
    def ddstor_torq__dbiasdbias(self, command: float, q: np.ndarray, os: Orbital_State) -> np.ndarray:
        if self.bias:
            return self.ddstor_torq__dudu(command=command, q=q, os=os)
        else:
            return np.zeros((0, 0, 0))
        
    def ddstor_torq__dbiasdbasestate(self, command: float, q: np.ndarray, os: Orbital_State) -> np.ndarray:
        if self.bias:
            return self.ddstor_torq__dudbasestate(command=command, q=q, os=os)
        else:
            return np.zeros((0, 7, 0))
        
    def ddstor_torq__dbiasdh(self, command: float, q: np.ndarray, os: Orbital_State) -> np.ndarray:
        if self.bias:
            return self.ddstor_torq__dudh(command=command, q=q, os=os)
        else:
            return np.zeros((0, 0, 0))
        
    def ddstor_torq__dbasestatedbasestate(self, command: float, q: np.ndarray, os: Orbital_State) -> np.ndarray:
        return np.zeros((7, 7, 0))
    
    def ddstor_torq__dbasestatedh(self, command: float, q: np.ndarray, os: Orbital_State) -> np.ndarray:
        return np.zeros((7, 0, 0))
    
    def ddstor_torq__dhdh(self, command: float, q: np.ndarray, os: Orbital_State) -> np.ndarray:
        return np.zeros((0, 0, 0))
    
        
    


    
