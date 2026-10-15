__all__ = ["Vector_Goal"]

import numpy as np
from typing import Tuple

from .goal import Goal
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.helpers.math_helpers import normalize, rot_mat, norm

class Vector_Goal(Goal):
    r"""
    Abstract base class for vector-alignment pointing goals.

    This class specializes :class:`~ADCS.goals.goal.Goal` for goals that
    align a specified spacecraft body-frame vector with a desired
    inertial-frame direction. Typical examples include aligning a
    sensor boresight with a target vector expressed in the ECI frame.

    Subclasses define a desired inertial direction
    :math:`\mathbf{v}_{\mathrm{ref}} \in \mathbb{R}^3` as a function of
    the orbital state. The reference mapping takes the form

    .. math::

        G_{\mathrm{vec}}(\mathcal{O}(t)) =
        \left(
            \mathbf{v}_{\mathrm{ref}},
            \boldsymbol{\omega}_{\mathrm{ref}}
        \right)

    where :math:`\boldsymbol{\omega}_{\mathrm{ref}}` is typically zero
    or defined by the subclass.

    This class provides a common geometric error computation based on
    quaternion representations of vector alignment.

    Subclasses must implement :meth:`to_ref`.

    See Also
    --------
    :class:`~ADCS.goals.goal.Goal`
    :class:`~ADCS.goals.attitude_goal.Attitude_Goal`

    """
    def __init__(self, boresight_name: str | None = None) -> None:
        r"""
        Initialize a vector-alignment goal.

        This constructor stores the optional boresight name to be used
        when computing the error with respect to the satellite's boresight
        dictionary.

        :param boresight_name:
            Optional name of the boresight to use from the satellite's
            boresight dictionary. If ``None``, the first available boresight
            is selected.
        :type boresight_name:
            str | None

        :return:
            None
        :rtype:
            None

        """
        super().__init__()
        self.boresight_name = boresight_name

    def to_ref(self, os0: Orbital_State) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Generate a reference inertial direction and angular velocity.

        Subclasses must implement this method to compute a desired
        inertial reference vector

        .. math::

            \mathbf{v}_{\mathrm{ref}} \in \mathbb{R}^3

        based on the current orbital state. The returned vector is
        expected to be nonzero and will typically be normalized by the
        downstream error computation.

        :param os0:
            Current orbital state.
        :type os0:
            Orbital_State

        :return:
            Reference inertial direction and reference angular velocity.
        :rtype:
            Tuple[numpy.ndarray, numpy.ndarray]

        """
        pass
    
    def error(self, q: np.ndarray, body_boresight: np.ndarray, os0: Orbital_State) -> np.ndarray:
        r"""
        Compute the vector-alignment attitude error.

        Let :math:`\mathbf{v}_{b}` be the normalized boresight vector
        expressed in the spacecraft body frame, and let
        :math:`\mathbf{v}_{\mathrm{ref}}` be the desired inertial
        direction returned by :meth:`to_ref`.

        The reference direction is transformed into the body frame using
        the direction cosine matrix derived from the current attitude
        quaternion :math:`\mathbf{q}`:

        .. math::

            \mathbf{v}_{\mathrm{ref}}^{b}
            = \mathbf{R}(\mathbf{q})^{\mathsf{T}}
              \mathbf{v}_{\mathrm{ref}}

        A quaternion representing the shortest rotation that aligns
        :math:`\mathbf{v}_{b}` with
        :math:`\mathbf{v}_{\mathrm{ref}}^{b}` is then constructed.

        For the nominal case, the error quaternion is

        .. math::

            \mathbf{q}_{\mathrm{err}} =
            \frac{1}{\sqrt{2(1 + d)}}
            \begin{bmatrix}
            1 + d \\
            \mathbf{v}_{\mathrm{ref}}^{b} \times \mathbf{v}_{b}
            \end{bmatrix}

        where

        .. math::

            d = \mathbf{v}_{b}^{\mathsf{T}}
                \mathbf{v}_{\mathrm{ref}}^{b}

        In the singular case :math:`d \approx -1`, corresponding to a
        180-degree misalignment, an arbitrary orthogonal rotation axis
        is selected.

        The returned error vector is the vector part of the error
        quaternion, with sign chosen to enforce the shortest rotation.

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
            Vector-alignment attitude error.
        :rtype:
            numpy.ndarray

        """
        eci_goal, _ = self.to_ref(os0)

        v_bore = normalize(body_boresight)
        R_b2i = rot_mat(q)                    # q: body -> ECI (Hamilton)
        v_goal_body = normalize(R_b2i.T @ eci_goal[1:4])

        dot = np.dot(v_bore, v_goal_body)

        if dot < -0.9999:
            # 180° case: pick any orthogonal axis
            axis = np.cross(v_bore, [1.0, 0.0, 0.0])
            if norm(axis) < 1e-3:
                axis = np.cross(v_bore, [0.0, 1.0, 0.0])
            q_err_full = np.concatenate([[0.0], normalize(axis)])
        else:
            # NOTE: goal × bore, not bore × goal
            cross = np.cross(v_goal_body, v_bore)
            q_err_full = normalize(np.concatenate([[1.0 + dot], cross]))

        sign = 1.0 if q_err_full[0] >= 0.0 else -1.0
        q_err_vec = q_err_full[1:4] * sign
        return q_err_vec
