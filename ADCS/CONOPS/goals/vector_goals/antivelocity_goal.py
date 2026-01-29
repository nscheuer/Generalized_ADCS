__all__ = ["AntiVelocity_Goal"]

import numpy as np
from typing import Tuple

from ADCS.CONOPS.goals import Vector_Goal
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.helpers.math_helpers import normalize

class AntiVelocity_Goal(Vector_Goal): 
    r"""
    Anti-velocity (ram-opposed) vector goal.

    This goal commands alignment opposite the instantaneous inertial velocity direction:

    .. math::

        \hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|}, \qquad
        \mathbf{r}_{goal} = -\hat{\mathbf{v}}.

    A practical feed-forward reference angular velocity is provided using the dominant
    orbital angular rate:

    .. math::

        \boldsymbol{\omega}_{ref}
        \approx
        \frac{\mathbf{r}\times\mathbf{v}}{\|\mathbf{r}\|^2}.
    """
    def to_ref(self, os0: Orbital_State) -> Tuple[np.ndarray, np.ndarray]:
        r = os0.R
        v = os0.V

        v_hat = normalize(v)
        w_ref = np.cross(r, v) / np.dot(r, r)

        r_ref = np.empty((4,))
        r_ref[0] = np.nan
        r_ref[1:] = -v_hat

        return r_ref, w_ref