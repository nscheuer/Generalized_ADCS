__all__ = ["No_Goal"]

import numpy as np
from typing import Tuple

from .goal import Goal
from ADCS.orbits.orbital_state import Orbital_State

class No_Goal(Goal):
    """
    Null goal representing the absence of a pointing objective.

    The :class:`No_Goal` class represents an explicit *do-nothing* goal.
    It is used when no valid guidance or pointing objective is active.

    This goal returns zero reference vectors and is commonly used as a
    safe fallback in scheduling, timeline, or initialization logic
    (e.g. when a :class:`~ADCS.goals.goal_list.GoalList` is empty).

    Unlike the base :class:`~ADCS.goals.goal.Goal`, this class explicitly
    encodes the semantics of “no objective” and should be preferred over
    implicit defaults.

    See Also
    --------
    :class:`~ADCS.goals.goal.Goal`
    :class:`~ADCS.goals.goal_list.GoalList`
    """
    def __init__(self) -> None:
        """
        Initialize a null goal.

        This constructor performs no initialization and exists solely to
        provide a concrete instance of a no-op goal.
        """
        pass

    def to_ref(self, os0: Orbital_State) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return zero inertial reference vectors.

        This method returns zero vectors for both the inertial pointing
        direction and the reference angular velocity, indicating that no
        guidance objective is currently active.

        Parameters
        ----------
        os0 : Orbital_State
            Current orbital state, provided as an
            :class:`~ADCS.orbits.orbital_state.Orbital_State`.
            This argument is unused.

        Returns
        -------
        r_goal_eci : numpy.ndarray, shape (3,)
            Zero vector.
        w_ref_eci : numpy.ndarray, shape (3,)
            Zero angular velocity vector.

        Notes
        -----
        This goal is intentionally inert and should not produce any motion
        or tracking behavior when used by downstream controllers.

        See Also
        --------
        :meth:`ADCS.goals.goal.Goal.to_ref`
        """
        zeros = np.array([0, 0, 0])
        return zeros, zeros
    
    def error(self, q: np.ndarray, body_boresight: np.ndarray, os0: Orbital_State) -> np.ndarray:
        return np.zeros(3)