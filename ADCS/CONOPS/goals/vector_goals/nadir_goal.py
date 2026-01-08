import numpy as np
from typing import Tuple

from ADCS.CONOPS.goals import Vector_Goal
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.helpers.math_helpers import normalize

class Nadir_Goal(Vector_Goal): 
    def to_ref(self, os0: Orbital_State) -> Tuple[np.ndarray, np.ndarray]:
        r = os0.R
        v = os0.V

        r_hat = normalize(r)
        r_goal = -r_hat
        w_ref = np.cross(r, v) / np.dot(r, r)

        return r_goal, w_ref
    