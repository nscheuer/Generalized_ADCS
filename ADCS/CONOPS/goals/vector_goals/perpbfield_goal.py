__all__ = ["PerpBField_Goal"]

import numpy as np
from typing import Tuple

from ADCS.CONOPS.goals import Vector_Goal
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.helpers.math_helpers import normalize

class PerpBField_Goal(Vector_Goal): 
    r"""
    Perpendicular-to-B-field vector goal.

    This goal commands alignment with a direction perpendicular to the local geomagnetic
    field in the inertial frame. A common choice is to form a perpendicular direction
    using the cross product with the velocity direction:

    .. math::

        \hat{\mathbf{B}} = \frac{\mathbf{B}_{ECI}}{\|\mathbf{B}_{ECI}\|}, \qquad
        \hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|}, \qquad
        \mathbf{r}_{goal} = \frac{\hat{\mathbf{B}} \times \hat{\mathbf{v}}}
                                {\|\hat{\mathbf{B}} \times \hat{\mathbf{v}}\|}.

    This direction is often useful for magnetorquer-only control since magnetic torque
    authority satisfies :math:`\boldsymbol{\tau} = \mathbf{m}\times\mathbf{B}` and is
    maximized for commands perpendicular to :math:`\mathbf{B}`.

    A practical feed-forward reference angular velocity may be approximated using the
    dominant orbital angular rate:

    .. math::

        \boldsymbol{\omega}_{ref}
        \approx
        \frac{\mathbf{r}\times\mathbf{v}}{\|\mathbf{r}\|^2}.
    """
    def to_ref(self, os0: Orbital_State) -> Tuple[np.ndarray, np.ndarray]:
        B = normalize(os0.B)
        v = normalize(os0.V)

        perp = np.cross(B, v)
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(B, [1.0, 0.0, 0.0])
            if np.linalg.norm(perp) < 1e-6:
                perp = np.cross(B, [0.0, 1.0, 0.0])

        perp_hat = normalize(perp)

        # Approximate angular rate = orbital rate
        w_ref = np.cross(os0.R, os0.V) / np.dot(os0.R, os0.R)

        return perp_hat, w_ref
    