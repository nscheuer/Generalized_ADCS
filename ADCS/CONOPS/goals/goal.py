__all__ = ["Goal"]

import numpy as np
from typing import Tuple

from ADCS.orbits.orbital_state import Orbital_State

class Goal:
    def __init__(self):
        pass

    def to_ref(self, x_hat: np.ndarray, os0: Orbital_State) -> Tuple[np.ndarray, np.ndarray]:
        return (np.array([0, 0, 1]), np.array([0, 0, 0]))