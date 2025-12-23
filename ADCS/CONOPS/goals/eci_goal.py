__all__ = ["ECI_Goal"]

import numpy as np
from typing import Tuple

from .goal import Goal
from ADCS.orbits.orbital_state import Orbital_State

class ECI_Goal(Goal):
    def __init__(self, eci_vector: np.ndarray) -> None:
        self.eci_vector = eci_vector

    def to_ref(self, os0: Orbital_State) -> Tuple[np.ndarray, np.ndarray]:
        eci_vector = self.eci_vector
        w_ref = np.array([0, 0, 0])
        return (eci_vector, w_ref)