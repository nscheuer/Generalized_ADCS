__all__ = ["Velocity_Goal"]

import numpy as np
from typing import Tuple

from ADCS.CONOPS.goals import Vector_Goal
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.helpers.math_helpers import normalize

class Velocity_Goal(Vector_Goal): 
    r"""
    Velocity-direction vector goal.

    This goal commands alignment with the instantaneous inertial velocity direction:

    .. math::

        \hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|}, \qquad
        \mathbf{r}_{goal} = \hat{\mathbf{v}}.

    A practical feed-forward reference angular velocity is provided using the dominant
    orbital angular rate (sufficient for most LEO ADCS applications):

    .. math::

        \boldsymbol{\omega}_{ref}
        \approx
        \frac{\mathbf{r}\times\mathbf{v}}{\|\mathbf{r}\|^2}.
    """
    def to_ref(self, os0: Orbital_State) -> Tuple[np.ndarray, np.ndarray]:
        r = os0.R
        v = os0.V

        v_hat = normalize(v)

        # Dominant angular rate of velocity direction
        w_ref = np.cross(r, v) / np.dot(r, r)

        return v_hat, w_ref