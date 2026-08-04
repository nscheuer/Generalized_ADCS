__all__ = ["ECI_Goal"]

import numpy as np
from typing import Tuple

from ADCS.CONOPS.goals import Vector_Goal
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.helpers.math_helpers import normalize

class ECI_Goal(Vector_Goal):
    r"""
    Fixed inertial pointing goal.

    The :class:`ECI_Goal` represents a pointing objective toward a
    fixed direction expressed directly in the Earth-Centered Inertial
    (ECI) frame. Unlike ground- or orbit-referenced goals, this goal
    does not depend on spacecraft position, velocity, or time.

    The reference angular velocity is identically zero, indicating
    that the target direction is inertially fixed.

    Parameters
    ----------
    eci_vector : numpy.ndarray
        Desired pointing direction expressed in the ECI frame.

    Attributes
    ----------
    eci_vector : numpy.ndarray
        Inertial reference direction vector.

    See Also
    --------
    :class:`~ADCS.goals.goal.Goal`
    :class:`~ADCS.orbits.orbital_state.Orbital_State`
    """
    def __init__(self, eci_vector: np.ndarray, boresight_name: str | None = None) -> None:
        r"""
        Initialize an inertial pointing goal.

        Parameters
        ----------
        eci_vector : numpy.ndarray
            Desired pointing direction expressed in the ECI frame.
            The vector is assumed to be constant in inertial space.

        boresight_name : str | None
            Optional name of the boresight to use from the satellite's
            boresight dictionary. If ``None``, the first available boresight
            is selected.
        """
        super().__init__(boresight_name=boresight_name)
        self.eci_vector = normalize(eci_vector)

    def to_ref(self, os0: Orbital_State) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Return inertial reference vectors for fixed ECI pointing.

        This method returns a constant inertial reference direction and
        a zero angular velocity vector. The orbital state is accepted
        only to satisfy the :class:`~ADCS.goals.goal.Goal` interface and
        is not used in the computation.

        Parameters
        ----------
        os0 : Orbital_State
            Current orbital state, provided as an
            :class:`~ADCS.orbits.orbital_state.Orbital_State`.
            This argument is unused.

        Returns
        -------
        r_goal_eci : numpy.ndarray, shape (3,)
            Inertially fixed reference direction in the ECI frame.
        w_ref_eci : numpy.ndarray, shape (3,)
            Zero reference angular velocity vector.

        Notes
        -----
        This goal corresponds to a pure inertial pointing command:

        .. math::

            \boldsymbol{\omega}_{ref} = \mathbf{0}

        No normalization or validation is performed on the input vector;
        the caller is responsible for ensuring appropriate scaling.

        See Also
        --------
        :meth:`ADCS.goals.goal.Goal.to_ref`
        """
        eci_vector = self.eci_vector
        w_ref = np.array([0, 0, 0])

        r_ref = np.empty((4,))
        r_ref[0] = np.nan
        r_ref[1:] = eci_vector

        return r_ref, w_ref