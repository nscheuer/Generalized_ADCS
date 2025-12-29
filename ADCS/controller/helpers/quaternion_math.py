r"""
quaternion_math.py
===================

Core quaternion and rotation math utilities for the ADCS framework.

This module provides quaternion algebra, rotation matrix conversions,
and associated differential operations such as Jacobians and Hessians.

All functions are standalone and compatible with onboard estimator
and simulator modules.

"""
import numpy as np
from ADCS.helpers.math_helpers import normalize, rot_mat, norm

def vector_alignment_error(q: np.ndarray, eci_goal: np.ndarray, body_boresight: np.ndarray) -> np.ndarray:
    r"""
    Compute a vector-form attitude alignment error between two directions.

    This function computes a *minimal vector attitude error* that aligns a
    body-fixed boresight vector with a desired inertial goal direction.
    The error is returned as a 3-vector suitable for use in feedback
    control laws (e.g. PD or LQR attitude controllers).

    The method:
    
    1. Transforms the inertial goal direction into the body frame using
       the current attitude quaternion.
    2. Computes the quaternion that rotates the goal vector onto the
       body boresight vector.
    3. Extracts the vector part of the error quaternion, enforcing the
       shortest-rotation convention.

    Parameters
    ----------
    q : numpy.ndarray, shape (4,)
        Current attitude quaternion representing the rotation from
        body frame to ECI frame (Hamilton convention).
    eci_goal : numpy.ndarray, shape (3,)
        Desired pointing direction expressed in the ECI frame.
    body_boresight : numpy.ndarray, shape (3,)
        Body-frame boresight direction to be aligned with the goal.

    Returns
    -------
    numpy.ndarray, shape (3,)
        Vector part of the alignment error quaternion, expressed in
        the body frame. This vector is zero when perfect alignment
        is achieved.

    Notes
    -----
    Let:

    * \(\mathbf{v}_b\) be the normalized body boresight vector
    * \(\mathbf{v}_g\) be the normalized goal direction expressed in the body frame

    The full error quaternion is computed as:

    \[
        \mathbf{q}_{err}
        =
        \begin{bmatrix}
            1 + \mathbf{v}_g^\top \mathbf{v}_b \\
            \mathbf{v}_g \times \mathbf{v}_b
        \end{bmatrix}
    \]

    followed by normalization.

    The returned error vector is:

    \[
        \mathbf{e}
        =
        \operatorname{sign}(q_{err,0}) \, \mathbf{q}_{err,1:3}
    \]

    which enforces the shortest-rotation convention.

    Special Case
    ------------
    If the vectors are nearly antiparallel (\(\mathbf{v}_g^\top \mathbf{v}_b \approx -1\)),
    the rotation is ill-defined. In this case, an arbitrary orthogonal axis
    is selected to represent a 180° rotation.

    This avoids numerical instability while preserving correct error
    magnitude.

    See Also
    --------
    rot_mat
    normalize

    References
    ----------
    * Markley, F. L., & Crassidis, J. L., *Fundamentals of Spacecraft
      Attitude Determination and Control*, Springer.
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