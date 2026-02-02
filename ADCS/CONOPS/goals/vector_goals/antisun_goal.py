__all__ = ["AntiSun_Goal"]

import numpy as np
from typing import Tuple

from ADCS.CONOPS.goals import Vector_Goal
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.helpers.math_helpers import normalize

class AntiSun_Goal(Vector_Goal):
    r"""
    Anti-Sun vector goal.

    This goal commands alignment opposite the spacecraft-to-Sun inertial direction
    :math:`\mathbf{s}_{ECI}`:

    .. math::

        \hat{\mathbf{s}} = \frac{\mathbf{s}_{ECI}}{\|\mathbf{s}_{ECI}\|}, \qquad
        \mathbf{r}_{goal} = -\hat{\mathbf{s}}.

    Over typical ADCS control horizons, the Sun direction is treated as inertially fixed,
    so the feed-forward reference angular velocity is set to zero:

    .. math::

        \boldsymbol{\omega}_{ref} = \mathbf{0}.
    """
    def to_ref(self, os0: Orbital_State) -> Tuple[np.ndarray, np.ndarray]:
        sun_vec = os0.get_sun_eci()
        anti_sun = -normalize(sun_vec)

        r_ref = np.empty((4,))
        r_ref[0] = np.nan
        r_ref[1:] = anti_sun

        w_ref = np.zeros(3)

        return r_ref, w_ref