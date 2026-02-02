__all__ = ["Attitude_Goal"]

import numpy as np
from typing import Tuple

from .goal import Goal
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.helpers.math_helpers import normalize, rot_mat, quat_mult, quat_inv

class Attitude_Goal(Goal):
    r"""
    Abstract base class for attitude-based pointing goals.

    This class specializes :class:`~ADCS.goals.goal.Goal` for goals whose
    primary output is a desired spacecraft attitude expressed as a
    reference quaternion. Subclasses define a desired reference attitude
    :math:`\mathbf{q}_{\mathrm{ref}}` as a function of the current orbital
    state.

    The reference mapping implemented by subclasses takes the form

    .. math::

        G_{\mathrm{att}}(\mathcal{O}(t)) =
        \left(
            \mathbf{q}_{\mathrm{ref}},
            \boldsymbol{\omega}_{\mathrm{ref}}
        \right)

    where :math:`\mathbf{q}_{\mathrm{ref}} \in \mathbb{R}^4` is a unit
    quaternion representing the desired body-to-inertial rotation and
    :math:`\boldsymbol{\omega}_{\mathrm{ref}} \in \mathbb{R}^3` is the
    corresponding reference angular velocity.

    This class additionally provides a common quaternion-based attitude
    error computation suitable for feedback control laws.

    Subclasses must implement :meth:`to_ref`.

    See Also
    --------
    :class:`~ADCS.goals.goal.Goal`
    :class:`~ADCS.goals.goal_list.GoalList`

    """
    def __init__(self) -> None:
        r"""
        Initialize an attitude goal.

        This constructor performs no initialization beyond invoking the
        base :class:`~ADCS.goals.goal.Goal` constructor.

        :return:
            None
        :rtype:
            None

        """
        super().__init__()

    def to_ref(self, os0: Orbital_State) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Generate a reference attitude and angular velocity.

        Subclasses must implement this method to compute a desired
        reference quaternion and angular velocity based on the current
        orbital state.

        This method is expected to return

        .. math::

            \mathbf{q}_{\mathrm{ref}} \in \mathbb{R}^4, \quad
            \boldsymbol{\omega}_{\mathrm{ref}} \in \mathbb{R}^3

        with :math:`\lVert \mathbf{q}_{\mathrm{ref}} \rVert = 1`.

        :param os0:
            Current orbital state.
        :type os0:
            Orbital_State

        :return:
            Reference attitude quaternion and reference angular velocity.
        :rtype:
            Tuple[numpy.ndarray, numpy.ndarray]

        """
        raise NotImplementedError("Choose a subclass for Attitude_Goal.")
    
    def error(self, q: np.ndarray, body_boresight: np.ndarray, os0: Orbital_State) -> np.ndarray:
        r"""
        Compute the quaternion-based attitude error vector.

        Let :math:`\mathbf{q}` denote the current spacecraft attitude
        quaternion and :math:`\mathbf{q}_{\mathrm{ref}}` the reference
        attitude returned by :meth:`to_ref`. The attitude error quaternion
        is defined as

        .. math::

            \mathbf{q}_{\mathrm{err}} =
            \mathbf{q}^{-1} \otimes \mathbf{q}_{\mathrm{ref}}

        where :math:`\otimes` denotes quaternion multiplication.

        To enforce the shortest-rotation convention, the quaternion is
        negated if its scalar component is negative:

        .. math::

            \mathbf{q}_{\mathrm{err}} \leftarrow
            -\mathbf{q}_{\mathrm{err}}
            \quad \text{if} \quad q_{\mathrm{err},0} < 0

        The returned error vector is the vector part of the error
        quaternion:

        .. math::

            \mathbf{e} =
            \begin{bmatrix}
            q_{\mathrm{err},1} &
            q_{\mathrm{err},2} &
            q_{\mathrm{err},3}
            \end{bmatrix}^{\mathsf{T}}

        This representation is commonly used for small-angle feedback
        control laws.

        The body-frame boresight argument is accepted for interface
        compatibility but is not used in this computation.

        :param q:
            Current spacecraft attitude quaternion.
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
            Vector part of the quaternion attitude error.
        :rtype:
            numpy.ndarray

        """
        q_ref, _ = self.to_ref(os0)

        # Body-frame attitude error quaternion.
        # We want q_err such that applying -q_err[1:4] as torque drives q toward q_ref.
        # The correct formulation is: q_err = q_ref^-1 * q
        # This represents the rotation FROM q_ref TO q, so negating it drives toward q_ref.
        q_err = quat_mult(quat_inv(q_ref), q)

        # shortest rotation
        if q_err[0] < 0.0:
            q_err = -q_err

        return q_err[1:4]
