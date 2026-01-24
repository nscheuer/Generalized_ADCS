__all__ = ["Goal"]

import numpy as np
from typing import Tuple

from ADCS.orbits.orbital_state import Orbital_State

class Goal:
    r"""
    Base class for guidance and pointing goals.

    This class defines the abstract interface for all guidance and
    pointing objectives in the ADCS framework. A goal represents a
    mapping from the current orbital state to a reference definition
    used by downstream guidance and control laws.

    Mathematically, a goal defines a function

    .. math::

        G : \mathcal{O} \rightarrow
        \left( \mathbf{r}_{\mathrm{ref}}, \boldsymbol{\omega}_{\mathrm{ref}} \right)

    where :math:`\mathcal{O}` denotes the orbital state space,
    :math:`\mathbf{r}_{\mathrm{ref}} \in \mathbb{R}^3` is an inertial
    reference direction, and
    :math:`\boldsymbol{\omega}_{\mathrm{ref}} \in \mathbb{R}^3` is the
    reference angular velocity expressed in the same inertial frame.

    This base class provides no concrete guidance logic. All subclasses
    must override the reference generation and error computation
    methods.

    See Also
    --------
    :class:`~ADCS.goals.eci_goal.ECI_Goal`
    :class:`~ADCS.goals.coordinate_goal.Coordinate_Goal`
    :class:`~ADCS.goals.goal_list.GoalList`

    """
    def __init__(self):
        r"""
        Initialize a goal instance.

        This constructor performs no initialization. It exists to
        enforce a uniform interface across all subclasses of
        :class:`~ADCS.goals.goal.Goal`.

        :return:
            None
        :rtype:
            None

        """
        pass

    def to_ref(self, os0: Orbital_State) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Generate inertial reference vectors from the current orbital state.

        This method defines the primary mapping implemented by a goal.
        Given the current orbital state :math:`\mathcal{O}(t)`, it
        returns a reference direction and reference angular velocity:

        .. math::

            \left( \mathbf{r}_{\mathrm{ref}},
            \boldsymbol{\omega}_{\mathrm{ref}} \right)
            = G\!\left( \mathcal{O}(t) \right)

        The exact physical interpretation of these quantities depends
        on the specific subclass implementation.

        This base method is abstract and must be overridden.

        :param os0:
            Current orbital state.
        :type os0:
            Orbital_State

        :return:
            Inertial reference direction and reference angular velocity.
        :rtype:
            Tuple[numpy.ndarray, numpy.ndarray]

        """
        raise NotImplementedError("Use a subclass of Goal")
    
    def error(self, q: np.ndarray, body_boresight: np.ndarray, os0: Orbital_State) -> np.ndarray:
        r"""
        Compute the pointing or attitude error associated with the goal.

        This method computes an error quantity suitable for use by a
        guidance or control law. In general, the error is a function of

        * the current attitude representation :math:`\mathbf{q}`
        * the spacecraft body-frame boresight vector
        * the current orbital state

        A typical error definition may take the form

        .. math::

            \mathbf{e} =
            f\!\left(
            \mathbf{q},
            \mathbf{b},
            \mathcal{O}(t)
            \right)

        where :math:`\mathbf{b}` denotes the body-frame boresight vector.

        This base method is abstract and must be overridden by all
        subclasses.

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
            Goal-specific error vector.
        :rtype:
            numpy.ndarray

        """
        raise NotImplementedError("Use a subclass of Goal")