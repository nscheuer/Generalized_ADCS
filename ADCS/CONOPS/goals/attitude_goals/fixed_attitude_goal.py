__all__ = ["Fixed_Attitude_Goal"]

import numpy as np
from typing import Tuple

from ADCS.CONOPS.goals import Attitude_Goal
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.helpers.math_helpers import normalize

class Fixed_Attitude_Goal(Attitude_Goal):
    def __init__(self, q_ref: np.ndarray) -> None:
        self.q_ref = normalize(q_ref)

    def to_ref(self, os0: Orbital_State) -> Tuple[np.ndarray, np.ndarray]:
        return self.q_ref, np.zeros(3)