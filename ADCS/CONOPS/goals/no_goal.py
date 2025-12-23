__all__ = ["No_Goal"]

import numpy as np
from typing import Tuple

from .goal import Goal
from ADCS.orbits.orbital_state import Orbital_State

class No_Goal(Goal):
    def __init__(self) -> None:
        pass

    def to_ref(self, os0: Orbital_State) -> Tuple[np.ndarray, np.ndarray]:
        zeros = np.array([0, 0, 0])
        return zeros, zeros