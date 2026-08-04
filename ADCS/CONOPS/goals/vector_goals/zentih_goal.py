__all__ = ["Zenith_Goal"]

import numpy as np
from typing import Tuple

from ADCS.CONOPS.goals import Vector_Goal
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.helpers.math_helpers import normalize

class Zenith_Goal(Vector_Goal): 
    r"""
    Zenith-pointing vector goal.

    This goal commands alignment with the local zenith direction, i.e. the unit vector
    from the Earth's center toward the spacecraft, expressed in the inertial (ECI) frame:

    .. math::

        \hat{\mathbf{r}} = \frac{\mathbf{r}}{\|\mathbf{r}\|}, \qquad
        \mathbf{r}_{goal} = +\hat{\mathbf{r}}.

    A feed-forward reference angular velocity is provided to track the rotating zenith
    direction in inertial space:

    .. math::

        \boldsymbol{\omega}_{ref}
        =
        \frac{\mathbf{r}\times\mathbf{v}}{\|\mathbf{r}\|^2}.
    """
    def to_ref(self, os0: Orbital_State) -> Tuple[np.ndarray, np.ndarray]:
        r = os0.R
        v = os0.V

        r_hat = normalize(r)
        w_ref = np.cross(r, v) / np.dot(r, r)

        r_ref = np.empty((4,))
        r_ref[0] = np.nan
        r_ref[1:] = r_hat

        return r_ref, w_ref
    