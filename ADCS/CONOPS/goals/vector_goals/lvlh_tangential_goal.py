__all__ = ["LVLH_Tangential_Goal"]

import numpy as np
from typing import Tuple

from ADCS.CONOPS.goals import Vector_Goal
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.helpers.math_helpers import normalize

class LVLH_Tangential_Goal(Vector_Goal): 
    r"""
    LVLH tangential (along-track) vector goal.

    This goal commands alignment with the tangential axis of the local orbital frame
    (LVLH/RTN). Using the ECI position :math:`\mathbf{r}` and velocity :math:`\mathbf{v}`,
    define:

    .. math::

        \hat{\mathbf{R}} = \frac{\mathbf{r}}{\|\mathbf{r}\|}, \qquad
        \hat{\mathbf{N}} = \frac{\mathbf{r}\times\mathbf{v}}{\|\mathbf{r}\times\mathbf{v}\|}, \qquad
        \hat{\mathbf{T}} = \hat{\mathbf{N}} \times \hat{\mathbf{R}}.

    The commanded inertial direction is:

    .. math::

        \mathbf{r}_{goal} = \hat{\mathbf{T}}.

    A feed-forward reference angular velocity is provided consistent with the orbital
    rotation rate:

    .. math::

        \boldsymbol{\omega}_{ref}
        =
        \frac{\mathbf{r}\times\mathbf{v}}{\|\mathbf{r}\|^2}.
    """
    def to_ref(self, os0: Orbital_State) -> Tuple[np.ndarray, np.ndarray]:
        r = os0.R
        v = os0.V

        r_hat = normalize(r)
        h_hat = normalize(np.cross(r, v))
        t_hat = normalize(np.cross(h_hat, r_hat))

        # Same orbital angular rate
        w_ref = np.cross(r, v) / np.dot(r, r)

        return t_hat, w_ref