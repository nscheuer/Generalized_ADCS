__all__ = ["AntiBField_Goal"]

import numpy as np
from typing import Tuple

from ADCS.CONOPS.goals import Vector_Goal
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.helpers.math_helpers import normalize

class AntiBField_Goal(Vector_Goal): 
    r"""
    Anti-magnetic field (anti B-field) vector goal.

    This goal commands alignment opposite the local geomagnetic field direction expressed
    in the inertial frame, :math:`\mathbf{B}_{ECI}`:

    .. math::

        \hat{\mathbf{B}} = \frac{\mathbf{B}_{ECI}}{\|\mathbf{B}_{ECI}\|}, \qquad
        \mathbf{r}_{goal} = -\hat{\mathbf{B}}.

    If a time derivative :math:`\dot{\mathbf{B}}_{ECI}` is available, a feed-forward angular
    rate can be formed as:

    .. math::

        \boldsymbol{\omega}_{ref}
        =
        \frac{\mathbf{B}_{ECI} \times \dot{\mathbf{B}}_{ECI}}{\|\mathbf{B}_{ECI}\|^2}.

    If :math:`\dot{\mathbf{B}}_{ECI}` is not available, :math:`\boldsymbol{\omega}_{ref}` may
    be set to :math:`\mathbf{0}`.
    """
    def to_ref(self, os0: Orbital_State) -> Tuple[np.ndarray, np.ndarray]:
        b_vec = os0.get_b_eci()
        anti_b = -normalize(b_vec)

        r_ref = np.empty((4,))
        r_ref[0] = np.nan
        r_ref[1:] = anti_b

        w_ref = np.zeros(3)

        return r_ref, w_ref
    