__all__ = ["Attitude_Goal"]

import numpy as np
from typing import Tuple

from .goal import Goal
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.helpers.math_helpers import normalize, rot_mat, quat_mult, quat_inv

class Attitude_Goal(Goal):
    def __init__(self) -> None:
        super().__init__()

    def to_ref(self, os0: Orbital_State) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Choose a subclass for Attitude_Goal.")
    
    def error(self, q: np.ndarray, body_boresight: np.ndarray, os0: Orbital_State) -> np.ndarray:
        q_ref, _ = self.to_ref(os0)

        q_err = quat_mult(q_ref, quat_inv(q))

        if q_err[0] < 0.0:
            q_err = -q_err

        return q_err[1:4]
