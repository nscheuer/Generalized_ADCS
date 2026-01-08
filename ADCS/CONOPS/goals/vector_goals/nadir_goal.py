__all__ = ["Nadir_Goal"]

import numpy as np
from typing import Tuple

from ADCS.CONOPS.goals import Vector_Goal
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.helpers.math_helpers import normalize

class Nadir_Goal(Vector_Goal): 
    r"""
    Nadir-pointing vector goal.

    This goal commands alignment with the local nadir direction, i.e. the unit vector
    from the spacecraft toward the Earth's center, expressed in the inertial (ECI) frame:

    .. math::

        \hat{\mathbf{r}} = \frac{\mathbf{r}}{\|\mathbf{r}\|}, \qquad
        \mathbf{r}_{goal} = -\hat{\mathbf{r}}.

    A feed-forward reference angular velocity is provided to track the rotating nadir
    direction in inertial space:

    .. math::

        \boldsymbol{\omega}_{ref}
        =
        \frac{\mathbf{r}\times\mathbf{v}}{\|\mathbf{r}\|^2}.

    Here :math:`\mathbf{r}` and :math:`\mathbf{v}` are the spacecraft ECI position and
    velocity.
    """
    def to_ref(self, os0: Orbital_State) -> Tuple[np.ndarray, np.ndarray]:
        r = os0.R
        v = os0.V

        r_hat = normalize(r)
        r_goal = -r_hat
        w_ref = np.cross(r, v) / np.dot(r, r)

        return r_goal, w_ref
    