__all__ = ["No_Goal"]

import numpy as np
from typing import Tuple

from .goal import Goal
from ADCS.orbits.orbital_state import Orbital_State

class No_Goal(Goal):
    r"""
    Null goal representing the absence of a pointing objective.

    This class provides an explicit representation of a null or inactive
    guidance objective. It implements a degenerate goal mapping that
    produces zero-valued reference quantities regardless of the orbital
    state.

    Mathematically, the reference mapping implemented by this class is

    .. math::

        G_{\varnothing}(\mathcal{O}(t)) =
        \left(
            \mathbf{0},
            \mathbf{0}
        \right)

    where :math:`\mathcal{O}(t)` denotes the current orbital state,
    :math:`\mathbf{0} \in \mathbb{R}^3` is the zero vector, and both the
    reference direction and reference angular velocity are identically
    zero.

    This class is typically used as a safe fallback when no valid goal
    is scheduled, for example when a
    :class:`~ADCS.goals.goal_list.GoalList` is empty or queried outside
    its defined timeline.

    See Also
    --------
    :class:`~ADCS.goals.goal.Goal`
    :class:`~ADCS.goals.goal_list.GoalList`

    """
    def __init__(self) -> None:
        r"""
        Initialize a null goal.

        This constructor performs no initialization. It exists to
        provide a concrete instance of a no-operation goal that can be
        returned or scheduled explicitly.

        :return:
            None
        :rtype:
            None

        """
        pass

    def to_ref(self, os0: Orbital_State) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Return zero inertial reference vectors.

        This method implements the null reference mapping

        .. math::

            \mathbf{r}_{\mathrm{ref}} = \mathbf{0}, \quad
            \boldsymbol{\omega}_{\mathrm{ref}} = \mathbf{0}

        indicating that no pointing direction and no angular motion are
        commanded.

        The orbital state argument is accepted for interface consistency
        but is not used in the computation.

        :param os0:
            Current orbital state.
        :type os0:
            Orbital_State

        :return:
            Zero reference direction and zero reference angular velocity.
        :rtype:
            Tuple[numpy.ndarray, numpy.ndarray]

        """
        return np.array([0, 0, 0, 0]), np.array([0, 0, 0])
    
    def error(self, q: np.ndarray, body_boresight: np.ndarray, os0: Orbital_State) -> np.ndarray:
        r"""
        Return a zero error vector.

        This method defines the error associated with the null goal as
        identically zero:

        .. math::

            \mathbf{e}_{\varnothing} = \mathbf{0}

        This ensures that downstream guidance or control laws receive no
        corrective signal when no active objective is defined.

        :param q:
            Current spacecraft attitude representation.
        :type q:
            numpy.ndarray

        :param body_boresight:
            Boresight direction expressed in the spacecraft body frame.
        :type body_boresight:
            numpy.ndarray

        :param os0:
            Current orbital state.
        :type os0:
            Orbital_State

        :return:
            Zero-valued error vector.
        :rtype:
            numpy.ndarray

        """
        return np.zeros(3)