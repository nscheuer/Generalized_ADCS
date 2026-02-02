r"""
quaternion_math.py
===================

Core quaternion and rotation math utilities for the ADCS framework.

This module provides quaternion algebra, rotation matrix conversions,
and associated differential operations such as Jacobians and Hessians.

All functions are standalone and compatible with onboard estimator
and simulator modules.

"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from ADCS.helpers.math_helpers import normalize, rot_mat, norm


def vector_alignment_error(
    q: NDArray[np.float64],
    eci_goal: NDArray[np.float64],
    body_boresight: NDArray[np.float64]
) -> NDArray[np.float64]:
    r"""
    Compute a minimal vector attitude error aligning a body boresight to an inertial goal.

    The function returns a 3D error vector suitable for feedback control. The error
    represents the shortest rotation that aligns a body-fixed boresight vector with
    a desired inertial-frame direction.

    The procedure is:

    .. math::

        \mathbf{v}_g^b = \mathbf{R}(q)^\top \mathbf{v}_g^{eci}

        \mathbf{q}_{err} =
        \begin{bmatrix}
            1 + \mathbf{v}_g^{b\top} \mathbf{v}_b \\
            \mathbf{v}_g^b \times \mathbf{v}_b
        \end{bmatrix}

    followed by normalization and extraction of the vector part with shortest-rotation
    sign enforcement.

    :param q:
        Attitude quaternion representing the rotation from body frame to ECI frame.

    :type q:
        numpy.ndarray

    :param eci_goal:
        Desired pointing direction expressed in the ECI frame.

    :type eci_goal:
        numpy.ndarray

    :param body_boresight:
        Body-frame boresight direction to be aligned with the goal.

    :type body_boresight:
        numpy.ndarray

    :return:
        Vector part of the attitude error quaternion in the body frame.

    :rtype:
        numpy.ndarray

    """
    v_bore = normalize(body_boresight)
    R_b2i = rot_mat(q)                    # q: body -> ECI (Hamilton)
    v_goal_body = normalize(R_b2i.T @ eci_goal)

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

    q_err_vec = q_err_full[1:4] * np.sign(q_err_full[0])
    return q_err_vec