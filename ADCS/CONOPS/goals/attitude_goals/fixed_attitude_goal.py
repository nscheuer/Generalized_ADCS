__all__ = ["Fixed_Attitude_Goal"]

import numpy as np
from typing import Tuple

from ADCS.CONOPS.goals import Attitude_Goal
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.helpers.math_helpers import normalize

class Fixed_Attitude_Goal(Attitude_Goal):
    r"""
    Fixed inertial attitude goal.

    This class implements a concrete attitude goal that commands a
    constant reference attitude, independent of the current orbital
    state. The desired attitude is provided at construction time as a
    reference quaternion and is held fixed for all times.

    Let the reference quaternion be denoted by

    .. math::

        \mathbf{q}_{\mathrm{ref}} \in \mathbb{R}^4,
        \quad \lVert \mathbf{q}_{\mathrm{ref}} \rVert = 1

    The goal mapping implemented by this class is

    .. math::

        G_{\mathrm{fixed}}(\mathcal{O}(t)) =
        \left(
            \mathbf{q}_{\mathrm{ref}},
            \mathbf{0}
        \right)

    where :math:`\mathcal{O}(t)` is the current orbital state and the
    reference angular velocity is identically zero.

    This goal is suitable for inertial-pointing modes such as sun-safe
    attitudes, communication attitudes, or commissioning configurations
    where no tracking motion is required.

    See Also
    --------
    :class:`~ADCS.CONOPS.goals.Attitude_Goal`
    :class:`~ADCS.goals.goal.Goal`

    """
    def __init__(self, q_ref: np.ndarray) -> None:
        r"""
        Initialize a fixed attitude goal.

        The provided quaternion is normalized to ensure it represents a
        valid rotation.

        :param q_ref:
            Desired reference attitude quaternion.
        :type q_ref:
            numpy.ndarray

        :return:
            None
        :rtype:
            None

        """
        self.q_ref = normalize(q_ref)

    def to_ref(self, os0: Orbital_State) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Return the fixed reference attitude and zero angular velocity.

        This method ignores the orbital state and always returns the
        constant reference attitude specified at initialization.

        .. math::

            \mathbf{q}_{\mathrm{ref}}(t) = \mathbf{q}_{\mathrm{ref}}, \quad
            \boldsymbol{\omega}_{\mathrm{ref}}(t) = \mathbf{0}

        :param os0:
            Current orbital state.
        :type os0:
            Orbital_State

        :return:
            Reference attitude quaternion and zero reference angular
            velocity.
        :rtype:
            Tuple[numpy.ndarray, numpy.ndarray]

        """
        return self.q_ref, np.zeros(3)