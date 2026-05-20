import numpy as np
import math
from numba import njit
import ADCS.orbits.universal_constants as uc
from typing import List


num_eps = uc.TimeConstants.num_eps
zeroquat = uc.DefaultStates.zeroquat


@njit(cache=True)
def _cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ],
        dtype=np.float64,
    )


@njit(cache=True)
def _quat_qdot(w: np.ndarray, q: np.ndarray) -> np.ndarray:
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
    w0 = w[0]
    w1 = w[1]
    w2 = w[2]
    return 0.5 * np.array(
        [
            -(w0 * q1 + w1 * q2 + w2 * q3),
            q0 * w0 + q2 * w2 - q3 * w1,
            q0 * w1 + q3 * w0 - q1 * w2,
            q0 * w2 + q1 * w1 - q2 * w0,
        ],
        dtype=np.float64,
    )


@njit(cache=True)
def _wdot_no_rw_kernel(
    w: np.ndarray, total_torque: np.ndarray, J: np.ndarray, invJ_noRW: np.ndarray
) -> np.ndarray:
    Jw = w @ J
    return (-_cross3(w, Jw) + total_torque) @ invJ_noRW


@njit(cache=True)
def _wdot_rw_kernel(
    w: np.ndarray,
    h: np.ndarray,
    total_torque: np.ndarray,
    J: np.ndarray,
    invJ_noRW: np.ndarray,
    rw_axes: np.ndarray,
) -> np.ndarray:
    coupled_momentum = w @ J + h @ rw_axes
    return (-_cross3(w, coupled_momentum) + total_torque) @ invJ_noRW


@njit(cache=True)
def _rw_hdot_kernel(
    u_rw: np.ndarray, wdot: np.ndarray, rw_axes: np.ndarray, rw_js: np.ndarray
) -> np.ndarray:
    # Equivalent to: u_rw - (wdot @ rw_axes.T) @ np.diagflat(rw_js)
    # for 1D vectors. Keep this form to avoid allocating a temporary diagonal matrix.
    return u_rw - (wdot @ rw_axes.T) * rw_js


# ===============================================================
# Quaternion Algebra and Rotation Matrices
# ===============================================================


def rot_mat(q: np.ndarray) -> np.ndarray:
    r"""
    Construct the direction cosine matrix (DCM) corresponding to a quaternion
    using the **Hamilton quaternion convention**.

    The returned matrix performs a rotation from the **body frame** to the
    **inertial (ECI) frame**, such that

    .. math::

        \mathbf{v}_{\mathrm{ECI}} = \mathbf{R}(\mathbf{q})\,\mathbf{v}_{\mathrm{body}}

    where the quaternion is defined as

    .. math::

        \mathbf{q} = \begin{bmatrix} q_0 & q_1 & q_2 & q_3 \end{bmatrix}^\mathsf{T}

    with scalar-first ordering.

    The rotation matrix is explicitly given by

    .. math::

        \mathbf{R}(\mathbf{q}) =
        \begin{bmatrix}
        q_0^2+q_1^2-q_2^2-q_3^2 & 2(q_1q_2-q_0q_3) & 2(q_1q_3+q_0q_2) \\
        2(q_1q_2+q_0q_3) & q_0^2-q_1^2+q_2^2-q_3^2 & 2(q_2q_3-q_0q_1) \\
        2(q_1q_3-q_0q_2) & 2(q_2q_3+q_0q_1) & q_0^2-q_1^2-q_2^2+q_3^2
        \end{bmatrix}

    :param q: Unit quaternion in Hamilton form ``[q0, q1, q2, q3]``.
    :type q: numpy.ndarray
    :return: Direction cosine matrix mapping body-frame vectors to inertial frame.
    :rtype: numpy.ndarray

    """
    q0, q1, q2, q3 = q
    return np.array([
        [q0**2 + q1**2 - q2**2 - q3**2, 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
        [2 * (q1 * q2 + q0 * q3), q0**2 - q1**2 + q2**2 - q3**2, 2 * (q2 * q3 - q0 * q1)],
        [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), q0**2 - q1**2 - q2**2 + q3**2]
    ])


@njit(cache=True)
def Wmat(q: np.ndarray) -> np.ndarray:
    r"""
    Compute the quaternion kinematic matrix used in first-order quaternion
    propagation.

    The quaternion derivative is expressed as

    .. math::

        \dot{\mathbf{q}} = \frac{1}{2}\,\mathbf{W}(\mathbf{q})\,\boldsymbol{\omega}

    where :math:`\boldsymbol{\omega}` is the angular velocity expressed in the
    body frame.

    The matrix :math:`\mathbf{W}(\mathbf{q})` is defined as

    .. math::

        \mathbf{W}(\mathbf{q}) =
        \begin{bmatrix}
        -\mathbf{q}_v^\mathsf{T} \\
        q_0\mathbf{I}_3 + [\mathbf{q}_v]_\times
        \end{bmatrix}

    with :math:`[\cdot]_\times` denoting the skew-symmetric operator.

    :param q: Quaternion in Hamilton convention.
    :type q: numpy.ndarray
    :return: Quaternion kinematic matrix.
    :rtype: numpy.ndarray

    """
    W = np.zeros((4, 3))
    qv = q[1:4]
    W[0, :] = -qv
    W[1:4, :] = q[0] * np.eye(3) + skewsym(qv)
    return W


def drotmatTvecdq(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    r"""
    Compute the Jacobian of a rotated vector with respect to the quaternion.

    This function evaluates

    .. math::

        \frac{\partial}{\partial \mathbf{q}}
        \left( \mathbf{R}^\mathsf{T}(\mathbf{q})\,\mathbf{v} \right)

    where :math:`\mathbf{R}(\mathbf{q})` is the DCM associated with the quaternion.

    The closed-form expression is

    .. math::

        2
        \begin{bmatrix}
        q_0\mathbf{v} - \mathbf{q}_v \times \mathbf{v} \\
        (\mathbf{q}_v \cdot \mathbf{v})\mathbf{I}
        - \mathbf{q}_v\mathbf{v}^\mathsf{T}
        + \mathbf{v}\mathbf{q}_v^\mathsf{T}
        - q_0[\mathbf{v}]_\times
        \end{bmatrix}

    This Jacobian is frequently used in attitude estimation and disturbance
    torque linearization.

    :param q: Quaternion in Hamilton form.
    :type q: numpy.ndarray
    :param v: Vector expressed in the inertial frame.
    :type v: numpy.ndarray
    :return: Jacobian of the rotated vector with respect to the quaternion.
    :rtype: numpy.ndarray

    """

    qv = q[1:]
    return 2 * np.vstack([
        q[0]*v - np.cross(qv, v),
        np.eye(3)*np.dot(qv, v) - np.outer(qv, v)
        + np.outer(v, qv) - q[0]*skewsym(v)
    ])


@njit(cache=True)
def skewsym(v: np.ndarray) -> np.ndarray:
    r"""
    Construct the skew-symmetric matrix associated with a 3D vector.

    The skew-symmetric operator satisfies

    .. math::

        [\mathbf{v}]_\times \mathbf{a} = \mathbf{v} \times \mathbf{a}

    and is defined as

    .. math::

        [\mathbf{v}]_\times =
        \begin{bmatrix}
        0 & -v_3 & v_2 \\
        v_3 & 0 & -v_1 \\
        -v_2 & v_1 & 0
        \end{bmatrix}

    :param v: Three-dimensional vector.
    :type v: numpy.ndarray
    :return: Skew-symmetric cross-product matrix.
    :rtype: numpy.ndarray

    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def ddrotmatTvecdqdq(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    r"""
    Compute the Hessian of a rotated vector with respect to the quaternion.

    This function evaluates the second derivative tensor

    .. math::

        \frac{\partial^2}{\partial \mathbf{q}^2}
        \left( \mathbf{R}^\mathsf{T}(\mathbf{q})\,\mathbf{v} \right)

    yielding a rank-3 tensor with dimensions :math:`4 \times 4 \times 3`.

    This quantity is required in second-order optimization, uncertainty
    propagation, and higher-order filters.

    :param q: Quaternion.
    :type q: numpy.ndarray
    :param v: Vector to be rotated.
    :type v: numpy.ndarray
    :return: Hessian tensor of the rotated vector with respect to the quaternion.
    :rtype: numpy.ndarray

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


def _normalize_legacy(v: np.ndarray) -> np.ndarray:
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
    if sn == 0.0:
        return v
    return v / sn


@njit(cache=True)
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
    total = 0.0
    for i in range(v.shape[0]):
        total += v[i] * v[i]
    return math.sqrt(total)


@njit(cache=True)
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
    out = np.empty(3, dtype=np.float64)
    scale = 1.0 / (1.0 + quat[0])
    out[0] = quat[1] * scale
    out[1] = quat[2] * scale
    out[2] = quat[3] * scale
    return out


@njit(cache=True)
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
    q = quat.copy()
    if abs(q[0]) < num_eps:
        q[0] = num_eps * np.sign(q[0])
        q = normalize(q)
    out = np.empty(3, dtype=np.float64)
    out[0] = q[1] / q[0]
    out[1] = q[2] / q[0]
    out[2] = q[3] / q[0]
    return out


@njit(cache=True)
def quat_to_vec3(quat: np.ndarray, mode: int = 0) -> np.ndarray:
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
    q = quat.copy()
    if mode == 6:
        if q[0] > 0.0:
            q *= np.sign(q[0])
        return 2 * quat_to_mrp(q)
    if mode == 5:
        return 2 * quat_to_mrp(q)
    if mode == 4:
        out = np.empty(3, dtype=np.float64)
        out[0] = q[1]
        out[1] = q[2]
        out[2] = q[3]
        return out
    if mode == 3:
        if q[0] > 0.0:
            q *= np.sign(q[0])
        out = np.empty(3, dtype=np.float64)
        out[0] = q[1]
        out[1] = q[2]
        out[2] = q[3]
        return out
    if mode == 2:
        return quat_to_cayley(q)
    if mode == 1:
        return quat_to_mrp(q)
    if abs(q[0]) > num_eps:
        q *= np.sign(q[0])
    return quat_to_mrp(q)


@njit(cache=True)
def vec3_to_quat(v3, mode):
    r"""
    Convert a 3-element attitude parameter vector into a quaternion according
    to a specified representation mode.

    This function implements the inverse mapping of
    :func:`~ADCS.attitude.quat_to_vec3`, supporting Modified Rodrigues
    Parameters (MRP), Cayley parameters, direct vector-part representations,
    and scaled variants.

    Depending on ``mode``, the conversion is defined as:

    .. math::

        \mathbf{q} =
        \begin{cases}
        \text{MRP}^{-1}(\tfrac{1}{2}\mathbf{v}), & \text{mode}=5,6 \\
        \begin{bmatrix}\sqrt{1-\|\mathbf{v}\|^2} \\ \mathbf{v}\end{bmatrix},
        & \text{mode}=3,4 \\
        \text{Cayley}^{-1}(\mathbf{v}), & \text{mode}=2 \\
        \text{MRP}^{-1}(\mathbf{v}), & \text{mode}=0,1
        \end{cases}

    Sign conventions are enforced where applicable to ensure a positive scalar
    component.

    :param v3: 3-element attitude parameter vector.
    :type v3: numpy.ndarray
    :param mode: Conversion mode selector.
    :type mode: int
    :return: Quaternion in Hamilton convention.
    :rtype: numpy.ndarray

    """
    if mode == 6:
        q = mrp_to_quat(v3 / 2.0)
        sq = np.sign(q[0])
        if np.abs(sq) > 0.0:
            q *= sq
        return q
    if mode == 5:
        return mrp_to_quat(v3 / 2.0)
    if mode == 4:
        q = np.empty(4, dtype=np.float64)
        q[0] = math.sqrt(max(0.0, 1.0 - norm(v3) ** 2.0))
        q[1] = v3[0]
        q[2] = v3[1]
        q[3] = v3[2]
        return q
    if mode == 3:
        q = np.empty(4, dtype=np.float64)
        q[0] = math.sqrt(max(0.0, 1.0 - norm(v3) ** 2.0))
        q[1] = v3[0]
        q[2] = v3[1]
        q[3] = v3[2]
        return q
    if mode == 2:
        return cayley_to_quat(v3)
    elif mode == 1:
        return mrp_to_quat(v3)
    else:
        q = mrp_to_quat(v3)
        sq = np.sign(q[0])
        if np.abs(sq) > 0.0:
            q *= sq
        return q
    

@njit(cache=True)
def mrp_to_quat(mrp):
    r"""
    Convert Modified Rodrigues Parameters (MRP) to a quaternion.

    The mapping is defined using the rotation angle

    .. math::

        \theta = 4\arctan\left(\frac{\|\boldsymbol{\sigma}\|}{2}\right)

    and unit rotation axis

    .. math::

        \hat{\mathbf{u}} = \frac{\boldsymbol{\sigma}}{\|\boldsymbol{\sigma}\|}

    yielding the quaternion

    .. math::

        \mathbf{q} =
        \begin{bmatrix}
        \cos(\theta/2) \\
        \hat{\mathbf{u}}\sin(\theta/2)
        \end{bmatrix}

    This formulation follows the standard aerospace definition
    (see NASA TR-19960035754).

    :param mrp: Modified Rodrigues Parameters.
    :type mrp: numpy.ndarray
    :return: Equivalent quaternion.
    :rtype: numpy.ndarray

    """
    out = np.empty(4, dtype=np.float64)
    s2 = np.dot(mrp, mrp)
    denom = 1.0 + s2
    out[0] = (1.0 - s2) / denom
    out[1] = 2.0 * mrp[0] / denom
    out[2] = 2.0 * mrp[1] / denom
    out[3] = 2.0 * mrp[2] / denom
    return out  # .reshape((4,1))


@njit(cache=True)
def cayley_to_quat(cly):
    r"""
    Convert Cayley parameters to a quaternion.

    Cayley parameters :math:`\mathbf{p}` are related to the quaternion by

    .. math::

        \mathbf{q} =
        \frac{1}{\sqrt{1+\|\mathbf{p}\|^2}}
        \begin{bmatrix}
        1 \\
        \mathbf{p}
        \end{bmatrix}

    This representation avoids trigonometric functions but is singular at
    rotations of :math:`\pm\pi`.

    :param cly: Cayley parameter vector.
    :type cly: numpy.ndarray
    :return: Quaternion corresponding to the Cayley parameters.
    :rtype: numpy.ndarray

    """
    out = np.empty(4, dtype=np.float64)
    denom = math.sqrt(1.0 + norm(cly) ** 2.0)
    out[0] = 1.0 / denom
    out[1] = cly[0] / denom
    out[2] = cly[1] / denom
    out[3] = cly[2] / denom
    return out


@njit(cache=True)
def _quat_mult_2(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p0, p1, p2, p3 = p
    q0, q1, q2, q3 = q
    out = np.empty(4, dtype=np.float64)
    out[0] = p0 * q0 - (p1 * q1 + p2 * q2 + p3 * q3)
    out[1] = p0 * q1 + q0 * p1 + (p2 * q3 - p3 * q2)
    out[2] = p0 * q2 + q0 * p2 + (p3 * q1 - p1 * q3)
    out[3] = p0 * q3 + q0 * p3 + (p1 * q2 - p2 * q1)
    return out


def quat_mult(p: np.ndarray, q: np.ndarray, *extra) -> np.ndarray:
    r"""
    Compute the Hamilton product of one or more quaternions.

    Given two quaternions :math:`\mathbf{p}` and :math:`\mathbf{q}`, the Hamilton
    product is defined as

    .. math::

        \mathbf{p} \otimes \mathbf{q} =
        \begin{bmatrix}
        p_0 q_0 - \mathbf{p}_v^\mathsf{T}\mathbf{q}_v \\
        p_0\mathbf{q}_v + q_0\mathbf{p}_v + \mathbf{p}_v \times \mathbf{q}_v
        \end{bmatrix}

    This operation composes rotations and is non-commutative.

    Multiple quaternions may be supplied and are multiplied in sequence.

    :param p: First quaternion.
    :type p: numpy.ndarray
    :param q: Second quaternion.
    :type q: numpy.ndarray
    :param extra: Additional quaternions to multiply sequentially.
    :type extra: tuple
    :return: Resulting quaternion product.
    :rtype: numpy.ndarray

    """
    result = _quat_mult_2(p, q)
    if isinstance(extra, np.ndarray):
        return _quat_mult_2(result, extra)
    elif isinstance(extra, (list, tuple)):
        if len(extra) > 1:
            if len(extra) == 4 and all(isinstance(j, (int, float, complex)) for j in extra):
                return _quat_mult_2(result, np.asarray(extra, dtype=np.float64))
            elif all(len(j) == 4 for j in extra):
                for item in extra:
                    result = _quat_mult_2(result, np.asarray(item, dtype=np.float64))
                return result
            else:
                raise ValueError("These do not all appear to be quaternions.")
        elif len(extra) == 1:
            return _quat_mult_2(result, np.asarray(extra[0], dtype=np.float64))
    return result


@njit(cache=True)
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
    if v.size != 3:
        raise ValueError("rot_exp expects a 3-element rotation vector")
    phi = norm(v)
    if phi == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    u = v / phi
    out = np.empty(4, dtype=np.float64)
    half_phi = phi / 2.0
    s = math.sin(half_phi)
    out[0] = math.cos(half_phi)
    out[1] = u[0] * s
    out[2] = u[1] * s
    out[3] = u[2] * s
    return out


@njit(cache=True)
def quat_inv(q: np.ndarray) -> np.ndarray:
    r"""
    Compute the inverse of a quaternion.

    For a quaternion :math:`\mathbf{q}`, the inverse is

    .. math::

        \mathbf{q}^{-1} = \frac{\mathbf{q}^*}{\|\mathbf{q}\|^2}

    where :math:`\mathbf{q}^*` denotes the conjugate quaternion.

    For unit quaternions, this reduces to the conjugate.

    :param q: Quaternion.
    :type q: numpy.ndarray
    :return: Inverse quaternion.
    :rtype: numpy.ndarray
    """
    q0 = q[0]
    qv0 = q[1]
    qv1 = q[2]
    qv2 = q[3]
    nq = norm(q)
    if nq > num_eps:
        scale = 1.0 / (nq ** 2)
    else:
        scale = 1.0
    out = np.empty(4, dtype=np.float64)
    out[0] = q0 * scale
    out[1] = -qv0 * scale
    out[2] = -qv1 * scale
    out[3] = -qv2 * scale
    return out


@njit(cache=True)
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
    out = np.empty(v.shape[0], dtype=np.float64)
    if sn == 0.0:
        for i in range(v.shape[0]):
            out[i] = v[i]
        return out
    for i in range(v.shape[0]):
        out[i] = v[i] / sn
    return out


@njit(cache=True)
def _normed_vec_jac(v: np.ndarray) -> np.ndarray:
    l = v.size
    normv = norm(v)
    if normv > num_eps:
        return np.eye(l) / normv - np.outer(v, v) / normv**3
    return np.eye(l)


@njit(cache=True)
def _normed_vec_jac_with_dv(v: np.ndarray, dv: np.ndarray) -> np.ndarray:
    return dv @ _normed_vec_jac(v)


def normed_vec_jac(v, dv=None):
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
    if dv is None:
        return _normed_vec_jac(v)
    return _normed_vec_jac_with_dv(v, dv)


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


@njit(cache=True)
def _vec_norm_jac(v: np.ndarray) -> np.ndarray:
    l = v.size
    normv = norm(v)
    if normv > num_eps:
        return v / normv
    return np.ones(l)


@njit(cache=True)
def _vec_norm_jac_with_dv(v: np.ndarray, dv: np.ndarray) -> np.ndarray:
    return dv @ _vec_norm_jac(v)


def vec_norm_jac(v: np.ndarray, dv=None) -> np.ndarray:
    r"""
    Compute the Jacobian of the Euclidean norm of a vector.

    For :math:`n = \|\mathbf{v}\|`, the gradient is

    .. math::

        \frac{\partial n}{\partial \mathbf{v}} =
        \frac{\mathbf{v}}{\|\mathbf{v}\|}

    When an external Jacobian ``dv`` is provided, the chain rule is applied.

    :param v: Input vector.
    :type v: numpy.ndarray
    :param dv: Optional external Jacobian.
    :type dv: numpy.ndarray or None
    :return: Jacobian of the vector norm.
    :rtype: numpy.ndarray

    """
    if dv is None:
        return _vec_norm_jac(v)
    return _vec_norm_jac_with_dv(v, dv)


@njit(cache=True)
def _vec_norm_hess(v: np.ndarray) -> np.ndarray:
    l = v.size
    normv = norm(v)
    return np.eye(l) / normv - np.outer(v, v) / normv**3.0


# NOT @njit: ``ddv`` is in general a 3-D tensor (shape ``(M, N, k)``) and
# ``dndv`` is the 1-D gradient of ``norm(v)`` (shape ``(k,)``). Numpy's
# matmul broadcasts ``(M, N, k) @ (k,) -> (M, N)`` cleanly, but numba's
# ``@`` does not implement the 3-D-by-1-D contraction (raises
# ``TypingError: '@': inputs must have compatible dimensions``). The
# matching plain-Python ``normed_vec_hess`` chain-rule helper is also
# left un-jitted for the same reason. ``_vec_norm_jac`` and
# ``_vec_norm_hess`` remain ``@njit``-compiled.
def _vec_norm_hess_with_chain(v: np.ndarray, dv: np.ndarray, ddv: np.ndarray) -> np.ndarray:
    dndv = _vec_norm_jac(v)
    ddndvdv = _vec_norm_hess(v)
    return dv @ ddndvdv @ dv.T + ddv @ dndv


def vec_norm_hess(v: np.ndarray, dv=None, ddv=None) -> np.ndarray:
    r"""
    Compute the Hessian of the Euclidean norm of a vector.

    The second derivative of :math:`\|\mathbf{v}\|` is

    .. math::

        \frac{\partial^2 \|\mathbf{v}\|}{\partial \mathbf{v}^2} =
        \frac{1}{\|\mathbf{v}\|}\mathbf{I}
        - \frac{\mathbf{v}\mathbf{v}^\mathsf{T}}{\|\mathbf{v}\|^3}

    External Jacobians and Hessians are propagated using the chain rule.

    :param v: Input vector.
    :type v: numpy.ndarray
    :param dv: Jacobian of the vector.
    :type dv: numpy.ndarray or None
    :param ddv: Hessian of the vector.
    :type ddv: numpy.ndarray or None
    :return: Hessian of the vector norm.
    :rtype: numpy.ndarray

    """
    if dv is None:
        if ddv is not None:
            raise ValueError("If Jacobian of v is None, Hessian must also be None")
        return _vec_norm_hess(v)
    else:
        if ddv is None:
            raise ValueError("If Jacobian of v is provided, Hessian must also be provided")
        return _vec_norm_hess_with_chain(v, dv, ddv)
    
def matrix_row_normalize(m: np.ndarray) -> np.ndarray:
    r"""
    Normalize each row of a matrix to unit Euclidean norm.

    Each row :math:`\mathbf{m}_i` is transformed as

    .. math::

        \hat{\mathbf{m}}_i =
        \frac{\mathbf{m}_i}{\|\mathbf{m}_i\|}

    This operation is commonly used for normalizing vector observations.

    :param m: Input matrix.
    :type m: numpy.ndarray
    :return: Row-normalized matrix.
    :rtype: numpy.ndarray

    """
    return m/np.expand_dims(matrix_row_norm(m), axis=1)

def matrix_row_norm(m: np.ndarray) -> np.ndarray:
    r"""
    Compute the Euclidean norm of each row of a matrix.

    The output vector satisfies

    .. math::

        n_i = \|\mathbf{m}_i\|

    for each row :math:`\mathbf{m}_i`.

    :param m: Two-dimensional input matrix.
    :type m: numpy.ndarray
    :return: Vector of row norms.
    :rtype: numpy.ndarray

    """
    if len(m.shape) != 2:
        raise ValueError("Not a 2D matrix")
    return np.linalg.norm(m, ord=2, axis=1)

def wahbas_svd(weights, body, inertial):
    r"""
    Solve Wahba's attitude determination problem using singular value decomposition.

    Wahba's problem minimizes the cost function

    .. math::

        J(\mathbf{R}) =
        \frac{1}{2} \sum_{i=1}^N w_i
        \left\| \mathbf{b}_i - \mathbf{R}\mathbf{r}_i \right\|^2

    where :math:`\mathbf{b}_i` are body-frame measurements and
    :math:`\mathbf{r}_i` are inertial-frame reference vectors.

    The optimal rotation matrix is obtained via the SVD of the attitude profile
    matrix

    .. math::

        \mathbf{B} = \sum_i w_i\,\mathbf{b}_i\mathbf{r}_i^\mathsf{T}

    and subsequently converted to a quaternion.

    :param weights: Scalar weights associated with each vector observation.
    :type weights: iterable
    :param body: Body-frame measurement vectors.
    :type body: iterable
    :param inertial: Inertial-frame reference vectors.
    :type inertial: iterable
    :return: Optimal attitude quaternion.
    :rtype: numpy.ndarray

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

@njit(cache=True)
def square_mat_sections(mat: np.ndarray, vals: np.ndarray):
    r"""
    Extract a square submatrix from a matrix using index selection.

    The returned matrix is formed by selecting rows and columns specified
    by ``vals``:

    .. math::

        \mathbf{M}_{\mathrm{sub}} = \mathbf{M}[\text{vals}, \text{vals}]

    This utility is frequently used for covariance sub-block extraction.

    :param mat: Input matrix.
    :type mat: numpy.ndarray
    :param vals: Index array defining the submatrix.
    :type vals: numpy.ndarray
    :return: Square submatrix.
    :rtype: numpy.ndarray

    """

    tmp = mat[vals,:]
    return tmp[:,vals]

@njit(cache=True)
def state_norm_jac(xk):
    r"""
    Construct the Jacobian of a state vector with quaternion normalization.

    The Jacobian is identity except for the quaternion components, which are
    replaced by the quaternion normalization Jacobian

    .. math::

        \frac{\partial \hat{\mathbf{q}}}{\partial \mathbf{q}}

    This is commonly used in extended Kalman filters with quaternion states.

    :param xk: State vector containing a quaternion in indices ``[3:7]``.
    :type xk: numpy.ndarray
    :return: State normalization Jacobian.
    :rtype: numpy.ndarray

    """
    l = xk.shape[0]
    q = xk[3:7]
    out = np.eye(l)
    out[3:7,3:7] = np.eye(4)/norm(q) - np.outer(q,q)/norm(q)**3
    return out

@njit(cache=True)
def quat_diff(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    r"""
    Compute the relative quaternion between two attitudes.

    The quaternion error is defined as

    .. math::

        \mathbf{q}_{\mathrm{err}} =
        \mathbf{q}_0^{-1} \otimes \mathbf{q}_1

    The result is normalized and forced to have a non-negative scalar part,
    ensuring a unique minimal-angle representation.

    :param q0: Reference quaternion.
    :type q0: numpy.ndarray
    :param q1: Target quaternion.
    :type q1: numpy.ndarray
    :return: Relative quaternion mapping ``q0`` to ``q1``.
    :rtype: numpy.ndarray

    """
    q0 = normalize(q0)
    q1 = normalize(q1)
    q_err = _quat_mult_2(quat_inv(q0), q1)
    if q_err[0] < 0:
        q_err = -q_err
    return normalize(q_err)

def limit(u, umax):
    r"""
    Apply element-wise saturation limits to a control vector.

    The saturation rule depends on the type of ``umax``:

    ================= ============================================
    ``umax`` type      Behavior
    ================= ============================================
    scalar             symmetric limit ``[-umax, umax]``
    array              symmetric element-wise limits
    ``(umin, umax)``   asymmetric limits
    ================= ============================================

    This function is typically used for actuator saturation modeling.

    :param u: Input vector or scalar.
    :type u: array-like
    :param umax: Saturation specification.
    :type umax: scalar, array-like, or tuple
    :return: Saturated output.
    :rtype: numpy.ndarray

    """
    u = np.asarray(u)

    # CASE 1: umax is a scalar → symmetric clip
    if np.isscalar(umax):
        umax = float(umax)
        return np.clip(u, -umax, umax)

    # CASE 2: umax is a tuple/list → may be (umin, umax)
    if isinstance(umax, (tuple, list)):
        if len(umax) != 2:
            raise ValueError("When umax is tuple/list, must be (umin, umax).")
        umin, umax_val = umax
        umin = np.asarray(umin)
        umax_val = np.asarray(umax_val)
        return np.clip(u, umin, umax_val)

    # CASE 3: umax is an array → symmetric elementwise
    umax = np.asarray(umax)
    return np.clip(u, -umax, umax)


def quat_to_euler(q: np.ndarray) -> np.ndarray:
    r"""
    Convert a quaternion to 3-2-1 (yaw-pitch-roll) Euler angles.

    The conversion is performed via the corresponding direction cosine matrix
    and yields

    .. math::

        \begin{aligned}
        \phi &= \arctan2(R_{32}, R_{33}) \\
        	heta &= -\arcsin(R_{31}) \\
        \psi &= \arctan2(R_{21}, R_{11})
        \end{aligned}

    The output angles are expressed in degrees.

    :param q: Quaternion in Hamilton convention.
    :type q: numpy.ndarray
    :return: Euler angles ``[roll, pitch, yaw]`` in degrees.
    :rtype: numpy.ndarray

    """
    R = rot_mat(q)
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = -np.arcsin(R[2, 0])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw]) * 180.0 / np.pi


@njit(cache=True)
def dcm_to_quat(R: np.ndarray) -> np.ndarray:
    r"""
    Convert a direction cosine matrix to a quaternion using a numerically
    stable algorithm.

    The method evaluates the matrix trace

    .. math::

        \mathrm{tr}(\mathbf{R}) = R_{11} + R_{22} + R_{33}

    and selects the computation branch that maximizes numerical stability
    (Sheppard's method).

    The resulting quaternion follows the Hamilton convention with scalar-first
    ordering.

    :param R: Direction cosine matrix.
    :type R: numpy.ndarray
    :return: Unit quaternion corresponding to the rotation matrix.
    :rtype: numpy.ndarray

    """
    tr = np.trace(R)
    out = np.empty(4, dtype=np.float64)

    if tr > 0.0:
        q0 = 0.5 * np.sqrt(1.0 + tr)
        out[0] = q0
        out[1] = (R[2, 1] - R[1, 2]) / (4.0 * q0)
        out[2] = (R[0, 2] - R[2, 0]) / (4.0 * q0)
        out[3] = (R[1, 0] - R[0, 1]) / (4.0 * q0)
        return normalize(out)

    if R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
        q1 = 0.5 * np.sqrt(1.0 + 2.0 * R[0, 0] - tr)
        out[0] = (R[2, 1] - R[1, 2]) / (4.0 * q1)
        out[1] = q1
        out[2] = (R[0, 1] + R[1, 0]) / (4.0 * q1)
        out[3] = (R[0, 2] + R[2, 0]) / (4.0 * q1)
    elif R[1, 1] >= R[2, 2]:
        q2 = 0.5 * np.sqrt(1.0 + 2.0 * R[1, 1] - tr)
        out[0] = (R[0, 2] - R[2, 0]) / (4.0 * q2)
        out[1] = (R[0, 1] + R[1, 0]) / (4.0 * q2)
        out[2] = q2
        out[3] = (R[1, 2] + R[2, 1]) / (4.0 * q2)
    else:
        q3 = 0.5 * np.sqrt(1.0 + 2.0 * R[2, 2] - tr)
        out[0] = (R[1, 0] - R[0, 1]) / (4.0 * q3)
        out[1] = (R[0, 2] + R[2, 0]) / (4.0 * q3)
        out[2] = (R[1, 2] + R[2, 1]) / (4.0 * q3)
        out[3] = q3

    return normalize(out)