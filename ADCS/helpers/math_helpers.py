r"""
quaternion_utils.py
===================

Core quaternion and rotation math utilities for the ADCS framework.

This module provides quaternion algebra, rotation matrix conversions,
and associated differential operations such as Jacobians and Hessians.

All functions are standalone and compatible with onboard estimator
and simulator modules.

"""

import numpy as np
import math
import ADCS.orbits.universal_constants as uc
from typing import List


num_eps = uc.TIME.num_eps
zeroquat = uc.DEFAULT.zeroquat


# ===============================================================
# Quaternion Algebra and Rotation Matrices
# ===============================================================

def rot_mat(q: np.ndarray) -> np.ndarray:
    r"""
    Compute the direction cosine matrix (rotation matrix) corresponding
    to a given quaternion, using the **Hamilton convention**.

    This defines the rotation from **body frame → inertial (ECI) frame**, so
    the returned matrix :math:`\mathbf{A}` transforms body-frame vectors
    into the inertial frame:

    .. math::
        \mathbf{v}_{\text{ECI}} = \mathbf{A}\,\mathbf{v}_{\text{body}}

    Conversely, to rotate from ECI to body frame, use :math:`\mathbf{A}^\top`.

    Parameters
    ----------
    q : numpy.ndarray, shape (4,)
        Quaternion in Hamilton form, :math:`[q_0, q_1, q_2, q_3]`, normalized.

    Returns
    -------
    A : numpy.ndarray, shape (3, 3)
        Direction cosine matrix mapping body-frame vectors to ECI frame.

    Notes
    -----
    - Quaternion follows Hamilton convention with scalar first.
    - :math:`q_0` is the scalar part, :math:`[q_1, q_2, q_3]` are the vector parts.
    """

    q0, q1, q2, q3 = q
    return np.array([
        [q0**2+q1**2-q2**2-q3**2, 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
        [2*(q1*q2+q0*q3), q0**2-q1**2+q2**2-q3**2, 2*(q2*q3-q0*q1)],
        [2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), q0**2-q1**2-q2**2+q3**2]
    ])


def Wmat(q: np.ndarray) -> np.ndarray:
    r"""
    Compute the quaternion kinematic matrix :math:`\mathbf{W}(\mathbf{q})`
    such that:

    .. math::
        \dot{\mathbf{q}} = \tfrac{1}{2}\,\mathbf{W}(\mathbf{q})\,\boldsymbol{\omega}

    where :math:`\boldsymbol{\omega}` is the angular velocity in body frame.

    Parameters
    ----------
    q : numpy.ndarray, shape (4,)
        Quaternion in Hamilton form.

    Returns
    -------
    W : numpy.ndarray, shape (4, 3)
        Quaternion kinematic matrix.

    Notes
    -----
    - The derivative satisfies :math:`\dot{\mathbf{q}} = \frac{1}{2}W(q)\omega`.
    - Consistent with the same quaternion convention as :func:`rot_mat`.
    """
    W = np.zeros((4, 3))
    qv = q[1:4]
    W[0, :] = -qv
    W[1:4, :] = q[0] * np.eye(3) + skewsym(qv)
    return W


def drotmatTvecdq(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    r"""
    Compute the derivative of the rotated vector
    :math:`\mathbf{v}_B = \mathbf{R}^\top(\mathbf{q}) \mathbf{v}`
    with respect to the quaternion :math:`\mathbf{q}`.

    .. math::
        \frac{\partial (\mathbf{R}^\top(\mathbf{q}) \mathbf{v})}{\partial \mathbf{q}}
        = 2
        \begin{bmatrix}
        q_0\,\mathbf{v} - \boldsymbol{q}_v \times \mathbf{v} \\
        (\boldsymbol{q}_v \cdot \mathbf{v})\mathbf{I}_3
        - \boldsymbol{q}_v \mathbf{v}^\top
        + \mathbf{v}\boldsymbol{q}_v^\top
        - q_0\,\text{skew}(\mathbf{v})
        \end{bmatrix}

    Parameters
    ----------
    q : numpy.ndarray, shape (4,)
        Quaternion in Hamilton form.
    v : numpy.ndarray, shape (3,)
        Vector to be rotated.

    Returns
    -------
    dRBTv__dq : numpy.ndarray, shape (4, 3)
        Partial derivative of :math:`\mathbf{R}^\top(\mathbf{q})\mathbf{v}` w.r.t. quaternion.

    Notes
    -----
    - Used for computing Jacobians of disturbance torques or forces.
    - :func:`skewsym` builds the skew-symmetric cross-product matrix.
    """
    qv = q[1:]
    return 2 * np.vstack([
        q[0]*v - np.cross(qv, v),
        np.eye(3)*np.dot(qv, v) - np.outer(qv, v)
        + np.outer(v, qv) - q[0]*skewsym(v)
    ])


def skewsym(v: np.ndarray) -> np.ndarray:
    r"""
    Return the skew-symmetric matrix corresponding to a 3D vector.

    .. math::
        \text{skew}(\mathbf{v}) =
        \begin{bmatrix}
        0 & -v_3 & v_2 \\
        v_3 & 0 & -v_1 \\
        -v_2 & v_1 & 0
        \end{bmatrix}

    Parameters
    ----------
    v : numpy.ndarray, shape (3,)
        3D vector.

    Returns
    -------
    S : numpy.ndarray, shape (3, 3)
        Skew-symmetric matrix such that :math:`S\mathbf{a} = \mathbf{v} \times \mathbf{a}`.
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def ddrotmatTvecdqdq(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    r"""
    Compute the second derivative (Hessian) of
    :math:`\mathbf{R}^\top(\mathbf{q})\mathbf{v}` with respect to the quaternion.

    Parameters
    ----------
    q : numpy.ndarray, shape (4,)
        Quaternion.
    v : numpy.ndarray, shape (3,)
        Vector to be rotated.

    Returns
    -------
    H : numpy.ndarray, shape (4, 4, 3)
        Hessian of rotated vector with respect to quaternion.
    """
    qv = q[1:]
    output = np.zeros((4, 4, 3))
    output[0, :, :] = 2 * np.vstack([v, -skewsym(v)])
    output[:, 0, :] = 2 * np.vstack([v, -skewsym(v)])
    tmp = 2 * np.multiply.outer(np.eye(3), v)
    output[1:, 1:, :] += -tmp
    output[1:, 1:, :] += np.transpose(tmp, (2, 0, 1))
    output[1:, 1:, :] += np.transpose(tmp, (1, 2, 0))
    return output


def normalize(v: np.ndarray) -> np.ndarray:
    r"""
    Normalize a vector to unit length.

    Parameters
    ----------
    v : numpy.ndarray
        Input vector.

    Returns
    -------
    v_norm : numpy.ndarray
        Normalized vector of same shape as input.

    Notes
    -----
    - If :math:`\|v\| = 0`, the vector is returned unchanged.
    """
    sn = norm(v)
    if sn == 0:
        return v
    return v / sn


def norm(v: np.ndarray) -> float:
    r"""
    Compute the Euclidean norm of a vector.

    Parameters
    ----------
    v : numpy.ndarray
        Input vector.

    Returns
    -------
    n : float
        Euclidean norm (magnitude) of the vector.
    """
    return np.linalg.norm(v)


def quat_to_mrp(quat: np.ndarray) -> np.ndarray:
    r"""
    Convert a quaternion to **Modified Rodrigues parameters (MRP)**.

    .. math::
        \boldsymbol{\sigma} = \frac{2\boldsymbol{q}_v}{1 + q_0}

    Parameters
    ----------
    quat : numpy.ndarray, shape (4,)
        Quaternion in Hamilton form.

    Returns
    -------
    sigma : numpy.ndarray, shape (3,)
        Modified Rodrigues parameters.
    """
    return 2 * quat[1:] / (1 + quat[0])


def quat_to_cayley(quat: np.ndarray) -> np.ndarray:
    r"""
    Convert a quaternion to **Cayley parameters**.

    .. math::
        \mathbf{p} = \frac{\boldsymbol{q}_v}{q_0}

    Parameters
    ----------
    quat : numpy.ndarray, shape (4,)
        Quaternion in Hamilton form.

    Returns
    -------
    p : numpy.ndarray, shape (3,)
        Cayley parameters.

    Notes
    -----
    - Ensures nonzero scalar part by applying a numerical epsilon if needed.
    """
    if abs(quat[0]) < num_eps:
        quat[0] = num_eps * np.sign(quat[0])
        quat = normalize(quat)
    return quat[1:] / quat[0]


def quat_to_vec3(quat: np.ndarray, mode: int) -> np.ndarray:
    r"""
    Convert a quaternion to a 3D attitude parameter vector according to mode.

    Parameters
    ----------
    quat : numpy.ndarray, shape (4,)
        Quaternion.
    mode : int
        Conversion mode:
        
        - 6 → 2×MRP (with positive scalar part)
        - 5 → 2×MRP
        - 4 → Vector component
        - 3 → Vector component (positive scalar)
        - 2 → Cayley parameters
        - 1 → MRP
        - 0 → Default: MRP (positive scalar)

    Returns
    -------
    v : numpy.ndarray, shape (3,)
        Equivalent attitude parameter vector.
    """
    if mode == 6:
        if quat[0] > 0.0:
            quat *= np.sign(quat[0])
        return 2 * quat_to_mrp(quat)
    if mode == 5:
        return 2 * quat_to_mrp(quat)
    if mode == 4:
        return quat[1:]
    if mode == 3:
        if quat[0] > 0.0:
            quat *= np.sign(quat[0])
        return quat[1:]
    if mode == 2:
        return quat_to_cayley(quat)
    if mode == 1:
        return quat_to_mrp(quat)
    if abs(quat[0]) > num_eps:
        quat *= np.sign(quat[0])
    return quat_to_mrp(quat)


def vec3_to_quat(v3,mode):
    if mode == 6:
        q = mrp_to_quat(v3/2.0)
        sq = np.sign(q[0])
        if np.abs(sq)>0.0:
            q *= sq
        return q
    if mode == 5:
        return mrp_to_quat(v3/2.0)
    if mode == 4:
        return np.concatenate([[math.sqrt(1.0-norm(v3)**2.0)],v3])
    if mode == 3:
        return np.concatenate([[math.sqrt(1.0-norm(v3)**2.0)],v3])
    if mode == 2:
        return cayley_to_quat(v3)
    elif mode == 1:
        return mrp_to_quat(v3)
    else:
        q = mrp_to_quat(v3)
        sq = np.sign(q[0])
        if np.abs(sq)>0.0:
            q *= sq
        return q
    

def mrp_to_quat(mrp):
    #https://ntrs.nasa.gov/api/citations/19960035754/downloads/19960035754.pdf
    # return (1/np.sqrt(1+norm(mrp)**2))*np.vstack([np.array([1]),mrp]).reshape((4,1))
    thetad2 = 2*math.atan(norm(mrp)*0.5)
    nhat = normalize(mrp)
    costd2 = math.cos(thetad2)
    return np.concatenate([[costd2],nhat*np.abs(math.sin(thetad2))])#.reshape((4,1))


def cayley_to_quat(cly):
    #https://ntrs.nasa.gov/api/citations/19960035754/downloads/19960035754.pdf
    # return (1/np.sqrt(1+norm(mrp)**2))*np.vstack([np.array([1]),mrp]).reshape((4,1))
    return np.concatenate([[1],cly])/np.sqrt(1+norm(cly)**2)


def quat_mult(p: np.ndarray, q: np.ndarray, *extra) -> np.ndarray:
    r"""
    Multiply one or more quaternions using the **Hamilton product**.

    Parameters
    ----------
    p, q : numpy.ndarray, shape (4,)
        Input quaternions.
    *extra : numpy.ndarray or list
        Additional quaternions to multiply.

    Returns
    -------
    pq : numpy.ndarray, shape (4,)
        Resulting quaternion from successive multiplications.

    Raises
    ------
    ValueError
        If input arguments are not valid quaternions.
    """
    if isinstance(extra, np.ndarray):
        return quat_mult(quat_mult(p, q), extra)
    elif isinstance(extra, (list, tuple)):
        if len(extra) > 1:
            if len(extra) == 4 and all(isinstance(j, (int, float, complex)) for j in extra):
                return quat_mult(quat_mult(p, q), extra[0])
            elif all(len(j) == 4 for j in extra):
                return quat_mult(quat_mult(p, q), extra[0], *extra[1:])
            else:
                raise ValueError("These do not all appear to be quaternions.")
        elif len(extra) == 1:
            return quat_mult(quat_mult(p, q), extra[0])
    p0, q0 = p[0], q[0]
    pv, qv = p[1:], q[1:]
    return np.concatenate([[p0*q0 - np.dot(pv, qv)], p0*qv + q0*pv + np.cross(pv, qv)])


def rot_exp(v: np.ndarray) -> np.ndarray:
    r"""
    Compute the exponential map :math:`\exp(\frac{\phi}{2}\hat{\mathbf{u}})`
    to obtain a quaternion from a rotation vector.

    Parameters
    ----------
    v : numpy.ndarray, shape (3,)
        Rotation vector :math:`\boldsymbol{\phi}` (axis × angle).

    Returns
    -------
    q : numpy.ndarray, shape (4,)
        Quaternion representing the rotation.

    Notes
    -----
    - For :math:`\|\boldsymbol{\phi}\| = 0`, returns identity quaternion.
    """
    assert v.size == 3
    phi = norm(v)
    if phi == 0:
        return zeroquat
    u = v / phi
    return np.concatenate([[math.cos(phi / 2)], u * math.sin(phi / 2)])


def quat_inv(q: np.ndarray) -> np.ndarray:
    r"""
    Compute the inverse (conjugate) of a quaternion.

    Parameters
    ----------
    q : numpy.ndarray, shape (4,)
        Quaternion.

    Returns
    -------
    q_inv : numpy.ndarray, shape (4,)
        Inverse quaternion :math:`q^{-1}`.
    """
    q0 = q[0]
    qv = q[-3:]
    nq = norm(q)
    if nq > num_eps:
        return np.concatenate([[q0], -qv]) / (nq**2)
    return np.concatenate([[q0], -qv])


def normalize(v: np.ndarray) -> np.ndarray:
    r"""
    Normalize a vector to unit length.

    Parameters
    ----------
    v : ndarray
        Input vector :math:`\mathbf{v}`.

    Returns
    -------
    ndarray
        Normalized vector
        :math:`\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|}`.
        If :math:`\|\mathbf{v}\|=0`, the zero vector is returned unchanged.
    """

    sn = norm(v)
    if sn == 0:
        return v
    return v/sn


def normed_vec_jac(v,dv=None):
    r"""
    Compute the Jacobian of the normalized vector :math:`\hat{\mathbf{v}}`.

    .. math::
        \frac{\partial \hat{\mathbf{v}}}{\partial \mathbf{v}}
        = \frac{1}{\|\mathbf{v}\|}\mathbf{I}
        - \frac{\mathbf{v}\mathbf{v}^\mathsf{T}}{\|\mathbf{v}\|^3}.

    Parameters
    ----------
    v : ndarray
        Vector :math:`\mathbf{v}` whose normalized derivative is taken.
    dv : ndarray, optional
        External Jacobian of :math:`\mathbf{v}` to be post-multiplied
        by the above term, if provided.

    Returns
    -------
    ndarray
        The Jacobian :math:`\partial \hat{\mathbf{v}}/\partial \mathbf{v}`
        (or ``dv @ dndv`` if ``dv`` is supplied).
    """
    l = v.size
    normv = norm(v)
    if normv>num_eps:
        dndv = np.eye(l)/normv - np.outer(v,v)/normv**3
    else:
        dndv = np.eye(l)
    if dv is None:
        return dndv
    return dv@dndv


def normed_vec_hess(v,dv=None,ddv=None):
    r"""
    Compute the Hessian of the normalized vector :math:`\hat{\mathbf{v}}`.

    The second derivative tensor satisfies

    .. math::
        \frac{\partial^2 \hat{\mathbf{v}}}{\partial v_i\,\partial v_j}
        = -\frac{v_i\,\mathbf{I}+v_j\,\mathbf{I}}{\|\mathbf{v}\|^3}
          + 3\,\frac{\mathbf{v}v_i v_j}{\|\mathbf{v}\|^5},

    which is symmetrized across indices.
    When external derivatives ``dv`` and ``ddv`` are provided,
    the chain rule is applied to yield the composed Hessian.

    Parameters
    ----------
    v : ndarray
        Vector :math:`\mathbf{v}`.
    dv : ndarray, optional
        Jacobian of :math:`\mathbf{v}` with respect to higher variables.
    ddv : ndarray, optional
        Hessian of :math:`\mathbf{v}` with respect to higher variables.

    Returns
    -------
    ndarray
        The Hessian tensor :math:`\partial^2 \hat{\mathbf{v}}/\partial \mathbf{v}^2`
        or its propagated form using ``dv``/``ddv``.
    """
    l = v.size
    normv = norm(v)
    dndv = np.eye(l)/normv - np.outer(v,v)/normv**3

    tmp = -np.multiply.outer(dndv,v/normv**2)
    ddndvdv = tmp + np.transpose(tmp,(2,0,1)) + np.transpose(tmp,(1,2,0))
    if dv is None:
        if ddv is not None:
            raise ValueError('if jacobian of v is none, hessian must also be none')
        return ddndvdv
    else:
        if ddv is None:
            raise ValueError('if jacobian of v is provided, hessian must also be provided')
        return np.tensordot(dv,dv@ddndvdv,([1],[0])) + ddv@dndv


def random_n_unit_vec(n: int) -> np.ndarray:
    r"""
    Generate a random unit-norm vector in :math:`\mathbb{R}^n`.

    The components are sampled from a standard normal distribution and normalized:

    .. math::
        \mathbf{v} = \frac{\mathbf{z}}{\|\mathbf{z}\|},
        \qquad  z_i \sim \mathcal{N}(0,1).

    Parameters
    ----------
    n : int
        Dimension of the vector space.

    Returns
    -------
    ndarray
        A random unit vector :math:`\mathbf{v}\in\mathbb{R}^n`
        with :math:`\|\mathbf{v}\|=1`.
    """
    return normalize(np.array([np.random.normal() for j in range(n)]))


def vec_norm_jac(v: np.ndarray, dv: np.ndarray = None) -> np.ndarray:
    l = v.size
    normv = norm(v)
    if normv > num_eps:
        dndv = v/normv
    else:
        dndv = np.ones(l)
    if dv is None:
        return dndv
    return dv@dndv


def vec_norm_hess(v: np.ndarray, dv: np.ndarray = None, ddv: np.ndarray = None) -> np.ndarray:
    l = v.size
    normv = norm(v)
    dndv = v/normv
    ddndvdv = np.eye(l)/normv - np.outer(v, v)/normv**3.0
    if dv is None:
        if ddv is not None:
            raise ValueError("If Jacobian of v is None, Hessian must also be None")
        return ddndvdv
    else:
        if ddv is None:
            raise ValueError("If Jacobian of v is provided, Hessian must also be provided")
        return dv@ddndvdv@dv.T + ddv@dndv
    
def matrix_row_normalize(m: np.ndarray) -> np.ndarray:
    return m/np.expand_dims(matrix_row_norm(m), axis=1)

def matrix_row_norm(m: np.ndarray) -> np.ndarray:
    if len(m.shape) != 2:
        raise ValueError("Not a 2D matrix")
    return np.linalg.norm(m, ord=2, axis=1)

def wahbas_svd(weights, body, inertial):
    """
    Solves Wahba's problem using SVD and returns a quaternion [q0, q1, q2, q3].
    """

    # Build attitude profile matrix B
    B = np.zeros((3,3))
    for w, b, r in zip(weights, body, inertial):
        B += w * np.outer(b, r)

    # SVD
    U, S, Vt = np.linalg.svd(B)
    M = np.eye(3)
    M[2,2] = np.linalg.det(U) * np.linalg.det(Vt)
    R = U @ M @ Vt   # rotation matrix

    # Convert rotation matrix → quaternion
    tr = np.trace(R)

    if tr > 0:
        q0 = 0.5 * np.sqrt(1 + tr)
        q1 = (R[2,1] - R[1,2]) / (4*q0)
        q2 = (R[0,2] - R[2,0]) / (4*q0)
        q3 = (R[1,0] - R[0,1]) / (4*q0)
        return np.array([q0, q1, q2, q3])

    # Otherwise, pick largest diagonal
    i = np.argmax([R[0,0], R[1,1], R[2,2]])

    if i == 0:
        q1 = 0.5*np.sqrt(1 + 2*R[0,0] - tr)
        q0 = (R[2,1] - R[1,2]) / (4*q1)
        q2 = (R[0,1] + R[1,0]) / (4*q1)
        q3 = (R[0,2] + R[2,0]) / (4*q1)

    elif i == 1:
        q2 = 0.5*np.sqrt(1 + 2*R[1,1] - tr)
        q0 = (R[0,2] - R[2,0]) / (4*q2)
        q1 = (R[0,1] + R[1,0]) / (4*q2)
        q3 = (R[1,2] + R[2,1]) / (4*q2)

    else:
        q3 = 0.5*np.sqrt(1 + 2*R[2,2] - tr)
        q0 = (R[1,0] - R[0,1]) / (4*q3)
        q1 = (R[0,2] + R[2,0]) / (4*q3)
        q2 = (R[1,2] + R[2,1]) / (4*q3)

    return np.array([q0, q1, q2, q3])

def square_mat_sections(mat: np.ndarray, vals: np.ndarray):
    tmp = mat[vals,:]
    return tmp[:,vals]

def state_norm_jac(xk):
    l = xk.shape[0]
    q = xk[3:7]
    out = np.eye(l)
    out[3:7,3:7] = quat_norm_jac(q)#np.eye(4)/norm(q) - np.outer(q,q)/norm(q)**3
    return out