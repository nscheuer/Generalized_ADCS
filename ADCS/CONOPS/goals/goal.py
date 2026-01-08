__all__ = ["Goal"]

import numpy as np
from typing import Tuple

from ADCS.orbits.orbital_state import Orbital_State

class Goal:
    """
    Base class for guidance and pointing goals.

    The :class:`Goal` class defines the common interface for all guidance
    and pointing objectives in the ADCS framework. Subclasses implement
    specific reference generation logic (e.g. inertial pointing,
    ground tracking, or time-varying objectives).

    A goal maps the current orbital state to:

    * an inertial reference direction
    * a reference angular velocity

    This base implementation provides a trivial default reference and
    is intended to be overridden by subclasses.

    See Also
    --------
    :class:`~ADCS.goals.eci_goal.ECI_Goal`
    :class:`~ADCS.goals.coordinate_goal.Coordinate_Goal`
    :class:`~ADCS.goals.goal_list.GoalList`
    """
    def __init__(self):
        """
        Initialize a goal instance.

        This base constructor performs no initialization and exists to
        provide a uniform interface across all goal types.
        """
        pass

    def to_ref(self, os0: Orbital_State) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate inertial reference vectors from the current orbital state.

        This method maps the spacecraft's orbital state to an inertial
        reference direction and a reference angular velocity.

        Subclasses must override this method to implement meaningful
        guidance behavior.

        Parameters
        ----------
        os0 : Orbital_State
            Current orbital state, provided as an
            :class:`~ADCS.orbits.orbital_state.Orbital_State`.

        Returns
        -------
        r_goal_eci : numpy.ndarray, shape (3,)
            Inertial reference direction in the ECI frame.
        w_ref_eci : numpy.ndarray, shape (3,)
            Reference angular velocity vector in the ECI frame.

        See Also
        --------
        :meth:`ADCS.goals.goal.Goal.to_ref`
        """
        raise NotImplementedError("Use a subclass of Goal")
    
    def error(self, q: np.ndarray, body_boresight: np.ndarray, os0: Orbital_State) -> np.ndarray:
        raise NotImplementedError("Use a subclass of Goal")