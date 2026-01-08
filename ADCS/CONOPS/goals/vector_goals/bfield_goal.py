__all__ = ["BField_Goal"]

import numpy as np
from typing import Tuple

from ADCS.CONOPS.goals import Vector_Goal
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.helpers.math_helpers import normalize

class BField_Goal(Vector_Goal): 
    r"""
    Magnetic field (B-field) vector goal.

    This goal commands alignment with the local geomagnetic field direction expressed in
    the inertial frame, :math:`\mathbf{B}_{ECI}`:

    .. math::

        \hat{\mathbf{B}} = \frac{\mathbf{B}_{ECI}}{\|\mathbf{B}_{ECI}\|}, \qquad
        \mathbf{r}_{goal} = \hat{\mathbf{B}}.

    If a time derivative :math:`\dot{\mathbf{B}}_{ECI}` is available (e.g. from a model or
    finite difference), a feed-forward angular rate can be formed as the angular velocity
    of the direction vector:

    .. math::

        \boldsymbol{\omega}_{ref}
        =
        \frac{\mathbf{B}_{ECI} \times \dot{\mathbf{B}}_{ECI}}{\|\mathbf{B}_{ECI}\|^2}.

    If :math:`\dot{\mathbf{B}}_{ECI}` is not available, :math:`\boldsymbol{\omega}_{ref}` may
    be set to :math:`\mathbf{0}` as a conservative default.
    """
    def to_ref(self, os0: Orbital_State) -> Tuple[np.ndarray, np.ndarray]:
        b_vec = os0.get_b_eci()

        w_ref = np.zeros(3)

        return normalize(b_vec), w_ref
    