"""Attitude determination from vector observations: TRIAD, QUEST and the q-method.

Given directions observed in the body frame (``body``) and the same directions
known in the inertial frame (``reference``), find the attitude. These are the
classic single-frame solutions of Wahba's problem and are used to warm-start
the recursive estimators from a first set of readings.

.. rubric:: Background

Most attitude sensors measure a direction: the magnetometers give the
direction of the magnetic field in the body frame, the sun sensors the
direction to the sun, a horizon sensor the direction to the Earth, a star
tracker the direction to a star. For each of these the same direction is also
known in the inertial frame, from the field model, the sun ephemeris, the
orbit, or the star catalogue. One such pair fixes two of the three attitude
angles; the rotation about the measured direction itself is invisible to it.
Two pairs that are not parallel fix the attitude completely, and more pairs
over-determine it.

Wahba [1]_ posed the problem of using all pairs at once: find the rotation
that maps the inertial directions onto the measured ones with the smallest
weighted sum of squared residuals, each weight being the inverse variance of
that observation's angular error. Everything in this module solves, or
approximates, that problem for one instant, without any dynamics or history,
which is why these are called single-frame or "point" methods:

* **TRIAD** [2]_ uses exactly two pairs. It builds an orthonormal triad from
  the two directions in each frame and reads the rotation off the two triads.
  It is exact and needs no iteration, but it trusts the first direction
  completely and uses the second only to fix the rotation about the first, so
  it is not the optimal combination of the two. Its covariance is obtained by
  linearising the construction (:func:`triad_covariance`).

* **The q-method** [3]_ (Davenport) notes that Wahba's loss is a quadratic
  form in the attitude quaternion, so the optimum is the eigenvector of a 4x4
  symmetric matrix built from the observations, belonging to its largest
  eigenvalue. It takes any number of pairs with any weights and is exact up to
  the eigen-solver.

* **QUEST** [4]_ (Shuster and Oh, "QUaternion ESTimator") reaches the same
  optimum without an eigen-solver, which is why most flight software uses it.
  The largest eigenvalue is close to the sum of the weights and is found by a
  Newton iteration on the characteristic polynomial; the quaternion then
  follows in closed form through the Gibbs vector, with the "sequential
  rotations" of the paper covering the 180-degree case where that vector is
  singular. The same paper gives the covariance of the optimal estimate, the
  small-angle attitude-error covariance ``[sum_i (1/sigma_i^2)(I - b_i b_i^T)]^-1``
  (:func:`wahba_covariance`), which is the best any estimator can do from
  those directions and is what makes the solution usable as the initial state
  and covariance of a filter.

QUEST and the q-method return the same attitude on exact data; here QUEST is
additionally checked against the q-method in the tests, and both are polished
by a few Gauss-Newton steps on the loss because the closed forms lose accuracy
when one direction is weighted far more heavily than the others (see
:func:`_refine`). Shepperd's method [5]_ is used wherever a quaternion has to
be read off a rotation matrix. Markley and Mortari [6]_ survey these and the
later algorithms, and Markley and Crassidis [7]_ give a textbook treatment.

.. rubric:: Why a single-frame solution is worth having

The recursive filters (EKF, MEKF, UKF) refine an attitude they are given; they
cannot start from nothing. Started at an arbitrary attitude with a large
covariance they spend their first part of a run converging, and because their
corrections are linearised about the current estimate, a start that is far
off can converge slowly, settle on a wrong solution, or diverge. A single-frame
solution from the first readings is accurate to the sensors' noise and comes
with a consistent covariance, so the filter begins where its linearisation is
valid. The same solution is useful on its own, as a sanity check of a filter
(a filter that drifts away from the point solution is diverging) and as a
plain attitude estimate when no dynamics model is wanted.

Conventions match the rest of the package: quaternions are Hamilton, scalar
first, and :func:`~ADCS.helpers.math_helpers.rot_mat` maps body to inertial,
so a unit reference direction ``r`` is observed as ``b = rot_mat(q).T @ r``.
Every function accepts one standard deviation per observation, in radians:
the angular error of that direction, assumed the same in every direction
perpendicular to it (the usual model behind QUEST).

References
----------

.. [1] G. Wahba, "A Least Squares Estimate of Satellite Attitude," *SIAM
   Review*, Vol. 7, No. 3, 1965, p. 409. doi:10.1137/1007077
.. [2] H. D. Black, "A Passive System for Determining the Attitude of a
   Satellite," *AIAA Journal*, Vol. 2, No. 7, 1964, pp. 1350-1351.
   doi:10.2514/3.2555
.. [3] P. B. Davenport, "A Vector Approach to the Algebra of Rotations with
   Applications," NASA Technical Note TN D-4696, 1968.
   https://ntrs.nasa.gov/citations/19680021122
.. [4] M. D. Shuster and S. D. Oh, "Three-Axis Attitude Determination from
   Vector Observations," *Journal of Guidance and Control*, Vol. 4, No. 1,
   1981, pp. 70-77. doi:10.2514/3.19717
.. [5] S. W. Shepperd, "Quaternion from Rotation Matrix," *Journal of
   Guidance and Control*, Vol. 1, No. 3, 1978, pp. 223-224.
   doi:10.2514/3.55767b
.. [6] F. L. Markley and D. Mortari, "Quaternion Attitude Estimation Using
   Vector Observations," *Journal of the Astronautical Sciences*, Vol. 48,
   No. 2-3, 2000, pp. 359-380.
.. [7] F. L. Markley and J. L. Crassidis, *Fundamentals of Spacecraft Attitude
   Determination and Control*, Springer, 2014, Chapter 5.
   doi:10.1007/978-1-4939-0802-8
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ADCS.helpers.math_helpers import rot_mat

__all__ = [
    "AttitudeSolution",
    "q_method",
    "quaternion_from_rotation_matrix",
    "quest",
    "triad",
    "triad_covariance",
    "wahba_covariance",
    "wahba_loss",
]

_PARALLEL_TOLERANCE = 1.0e-8


@dataclass(frozen=True)
class AttitudeSolution:
    """The result of a single-frame attitude solution.

    :param quaternion: Attitude quaternion (scalar first, ``rot_mat`` convention).
    :param covariance: 3x3 covariance of the small-angle attitude error in the
        body frame, in rad^2 (rotation-vector units), or ``None`` when no
        measurement uncertainty was given.
    :param loss: Wahba loss at the solution with weights normalised to sum
        to one; zero for perfect observations.
    :param method: ``"triad"``, ``"quest"`` or ``"q_method"``.
    """

    quaternion: np.ndarray
    covariance: np.ndarray | None
    loss: float
    method: str


# ----------------------------------------------------------------------------
# helpers


def _unit_rows(vectors: np.ndarray, name: str) -> np.ndarray:
    array = np.array(vectors, dtype=float, copy=True)
    if array.ndim == 1:
        array = array.reshape(1, 3)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n, 3), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms == 0.0):
        raise ValueError(f"{name} must not contain zero vectors")
    return array / norms[:, None]


def _sigmas(sigma: np.ndarray | float, count: int, name: str = "sigma") -> np.ndarray:
    array = np.broadcast_to(np.asarray(sigma, dtype=float), (count,)).copy()
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must be finite and positive")
    return array


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def quaternion_from_rotation_matrix(matrix: np.ndarray) -> np.ndarray:
    """Return the scalar-first quaternion ``q`` with ``rot_mat(q) == matrix``.

    Shepperd's method: the largest of the four squared components is computed
    first so no division by a small number occurs. The scalar part is made
    non-negative.
    """
    m = np.asarray(matrix, dtype=float)
    if m.shape != (3, 3):
        raise ValueError(f"matrix must be 3x3, got {m.shape}")
    trace = np.trace(m)
    candidates = np.array([trace, m[0, 0], m[1, 1], m[2, 2]])
    largest = int(np.argmax(candidates))
    if largest == 0:
        s = 2.0 * np.sqrt(1.0 + trace)
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    elif largest == 1:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif largest == 2:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    q = q / np.linalg.norm(q)
    return -q if q[0] < 0.0 else q


def _rotation_vector(matrix: np.ndarray) -> np.ndarray:
    """Rotation vector of a rotation matrix (exact, any angle below 180 degrees)."""
    cos_angle = np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cos_angle))
    axis_times_two_sin = np.array([matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]])
    if angle < 1.0e-8:
        return 0.5 * axis_times_two_sin
    return axis_times_two_sin * (angle / (2.0 * np.sin(angle)))


def attitude_error(quaternion: np.ndarray, reference_quaternion: np.ndarray) -> np.ndarray:
    """Body-frame rotation vector taking ``reference_quaternion`` to ``quaternion``.

    This is the right-error convention the estimators use (``dq = q_ref^-1 q``),
    so it is the quantity the returned covariances describe.
    """
    relative = rot_mat(reference_quaternion).T @ rot_mat(quaternion)
    return _rotation_vector(relative)


def _davenport_matrix(body: np.ndarray, reference: np.ndarray, weights: np.ndarray):
    """Return B, S, sigma, z of Davenport's K matrix for Wahba's problem."""
    attitude_profile = (weights[:, None] * body).T @ reference  # B = sum w b r^T
    symmetric = attitude_profile + attitude_profile.T
    trace = float(np.trace(attitude_profile))
    z = np.sum(weights[:, None] * np.cross(body, reference), axis=0)
    return attitude_profile, symmetric, trace, z


def _rotation_from_davenport_quaternion(vector: np.ndarray, scalar: float) -> np.ndarray:
    """Wahba's attitude matrix (reference to body) from Davenport's [vector; scalar] quaternion."""
    skew = np.array([[0.0, -vector[2], vector[1]], [vector[2], 0.0, -vector[0]], [-vector[1], vector[0], 0.0]])
    return (scalar**2 - vector @ vector) * np.eye(3) + 2.0 * np.outer(vector, vector) - 2.0 * scalar * skew


def _skew(vector: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -vector[2], vector[1]], [vector[2], 0.0, -vector[0]], [-vector[1], vector[0], 0.0]])


def _rotation_matrix(vector: np.ndarray) -> np.ndarray:
    """Rotation matrix of a rotation vector (Rodrigues); the inverse of :func:`_rotation_vector`."""
    angle = float(np.linalg.norm(vector))
    cross = _skew(vector)
    if angle < 1.0e-8:
        return np.eye(3) + cross + 0.5 * cross @ cross
    return np.eye(3) + np.sin(angle) / angle * cross + (1.0 - np.cos(angle)) / angle**2 * cross @ cross


def _refine(attitude: np.ndarray, body: np.ndarray, reference: np.ndarray, weights: np.ndarray, steps: int) -> np.ndarray:
    """Gauss-Newton steps on the Wahba cost, starting from ``attitude`` (reference -> body).

    The closed-form solutions (QUEST, q-method) lose accuracy when one
    direction is weighted much more heavily than the rest: the optimum is
    then an eigenvector whose eigenvalue almost coincides with the next one,
    and no eigen-solver locates such an eigenvector precisely (with weights
    1e6 apart QUEST was off by 3e-5 rad on exact data, with 1e8 by 6e-4). A
    Gauss-Newton step instead solves the linearised least-squares problem by
    orthogonal factorisation, where the weight ratio only enters as its
    square root, so a few steps recover the optimum to machine precision.
    """
    root_weights = np.sqrt(weights)
    for _ in range(int(steps)):
        predicted = reference @ attitude.T  # rows: attitude @ r_i, the directions the estimate expects in the body frame
        residual = ((body - predicted) * root_weights[:, None]).reshape(-1)
        jacobian = np.vstack([weight * _skew(direction) for weight, direction in zip(root_weights, predicted)])
        step = np.linalg.lstsq(jacobian, residual, rcond=None)[0]
        attitude = _rotation_matrix(step).T @ attitude
        if float(np.linalg.norm(step)) < 1.0e-14:
            break
    return attitude


def _loss(attitude: np.ndarray, body: np.ndarray, reference: np.ndarray, weights: np.ndarray) -> float:
    """Wahba loss of ``attitude`` (reference -> body) with weights that sum to one."""
    predicted = reference @ attitude.T
    return max(0.0, 1.0 - float(np.sum(weights * np.sum(body * predicted, axis=1))))


# ----------------------------------------------------------------------------
# loss and covariances


def wahba_loss(quaternion: np.ndarray, body: np.ndarray, reference: np.ndarray, sigma: np.ndarray | float) -> float:
    """Wahba loss ``0.5 * sum w_i |b_i - rot_mat(q).T r_i|^2`` with weights normalised to one."""
    body = _unit_rows(body, "body")
    reference = _unit_rows(reference, "reference")
    weights = 1.0 / _sigmas(sigma, body.shape[0]) ** 2
    weights = weights / np.sum(weights)
    predicted = reference @ rot_mat(np.asarray(quaternion, dtype=float))  # rows: rot_mat(q).T r_i
    return 0.5 * float(np.sum(weights * np.sum((body - predicted) ** 2, axis=1)))


def wahba_covariance(body: np.ndarray, sigma: np.ndarray | float) -> np.ndarray:
    """Covariance of the optimal attitude estimate for the given observations.

    ``P = [sum_i (1/sigma_i^2)(I - b_i b_i^T)]^-1`` (Shuster and Oh 1981): the
    small-angle attitude-error covariance in the body frame, in rad^2. It is
    the best any estimator can do from these directions. With a single
    direction, or directions that are all parallel, the rotation about that
    direction is unobservable and a ``ValueError`` is raised.
    """
    body = _unit_rows(body, "body")
    weights = 1.0 / _sigmas(sigma, body.shape[0]) ** 2
    information = np.zeros((3, 3))
    for direction, weight in zip(body, weights):
        information += weight * (np.eye(3) - np.outer(direction, direction))
    eigenvalues = np.linalg.eigvalsh(information)
    if eigenvalues[0] <= 1.0e-14 * eigenvalues[-1]:  # below the numerical resolution of the sum
        raise ValueError(
            "the attitude is not observable from these directions: the rotation about "
            "their common axis is undetermined (at least two non-parallel directions are needed)"
        )
    covariance = np.linalg.inv(information)
    return 0.5 * (covariance + covariance.T)


def triad_covariance(
    body_1: np.ndarray,
    body_2: np.ndarray,
    reference_1: np.ndarray,
    reference_2: np.ndarray,
    sigma_1: float,
    sigma_2: float,
) -> np.ndarray:
    """Covariance of the TRIAD solution, by linearising the construction.

    TRIAD trusts the first direction completely, so its error is larger than
    the optimal (:func:`wahba_covariance`) one; the covariance is obtained by
    propagating each observation's tangent-plane noise through the TRIAD map
    with central differences.
    """
    b1, b2 = _unit_rows(body_1, "body_1")[0], _unit_rows(body_2, "body_2")[0]
    r1, r2 = _unit_rows(reference_1, "reference_1")[0], _unit_rows(reference_2, "reference_2")[0]
    nominal = triad(b1, b2, r1, r2).quaternion
    step = 1.0e-6
    covariance = np.zeros((3, 3))
    for index, (direction, sigma) in enumerate(((b1, sigma_1), (b2, sigma_2))):
        jacobian = np.zeros((3, 3))
        for axis in range(3):
            delta = np.zeros(3)
            delta[axis] = step
            plus = [b1, b2]
            minus = [b1, b2]
            plus[index] = _unit(direction + delta)
            minus[index] = _unit(direction - delta)
            forward = attitude_error(triad(plus[0], plus[1], r1, r2).quaternion, nominal)
            backward = attitude_error(triad(minus[0], minus[1], r1, r2).quaternion, nominal)
            jacobian[:, axis] = (forward - backward) / (2.0 * step)
        tangent_noise = float(sigma) ** 2 * (np.eye(3) - np.outer(direction, direction))
        covariance += jacobian @ tangent_noise @ jacobian.T
    return 0.5 * (covariance + covariance.T)


# ----------------------------------------------------------------------------
# solvers


def triad(
    body_1: np.ndarray,
    body_2: np.ndarray,
    reference_1: np.ndarray,
    reference_2: np.ndarray,
    *,
    sigma_1: float | None = None,
    sigma_2: float | None = None,
) -> AttitudeSolution:
    """TRIAD: the attitude from two directions, trusting the first one exactly.

    The first direction is matched exactly and the second fixes the rotation
    about it, so put the more accurate direction first. Raises when the two
    directions are parallel (in either frame), where the rotation about them
    cannot be determined.
    """
    b1, b2 = _unit_rows(body_1, "body_1")[0], _unit_rows(body_2, "body_2")[0]
    r1, r2 = _unit_rows(reference_1, "reference_1")[0], _unit_rows(reference_2, "reference_2")[0]
    body_cross = np.cross(b1, b2)
    reference_cross = np.cross(r1, r2)
    if np.linalg.norm(body_cross) < _PARALLEL_TOLERANCE or np.linalg.norm(reference_cross) < _PARALLEL_TOLERANCE:
        raise ValueError("TRIAD needs two non-parallel directions")
    body_frame = np.column_stack([b1, _unit(body_cross), np.cross(b1, _unit(body_cross))])
    reference_frame = np.column_stack([r1, _unit(reference_cross), np.cross(r1, _unit(reference_cross))])
    body_to_inertial = reference_frame @ body_frame.T
    quaternion = quaternion_from_rotation_matrix(body_to_inertial)
    covariance = None
    if sigma_1 is not None and sigma_2 is not None:
        covariance = triad_covariance(b1, b2, r1, r2, float(sigma_1), float(sigma_2))
    loss = wahba_loss(quaternion, np.vstack([b1, b2]), np.vstack([r1, r2]),
                      np.array([1.0, 1.0]) if sigma_1 is None or sigma_2 is None else np.array([sigma_1, sigma_2]))
    return AttitudeSolution(quaternion=quaternion, covariance=covariance, loss=loss, method="triad")


def q_method(body: np.ndarray, reference: np.ndarray, sigma: np.ndarray | float, *, refine_steps: int = 8) -> AttitudeSolution:
    """Davenport's q-method: the exact Wahba optimum from the 4x4 eigenproblem.

    Slower than :func:`quest` in principle and free of special cases; QUEST
    is checked against it in the tests. Like QUEST it is polished by
    ``refine_steps`` Gauss-Newton steps, because the eigenvector of the 4x4
    matrix is itself imprecise when one direction dominates the weights.
    """
    body = _unit_rows(body, "body")
    reference = _unit_rows(reference, "reference")
    if body.shape != reference.shape:
        raise ValueError("body and reference must have the same shape")
    if body.shape[0] < 2:
        raise ValueError("at least two directions are needed to determine the attitude")
    sigmas = _sigmas(sigma, body.shape[0])
    weights = 1.0 / sigmas**2
    weights = weights / np.sum(weights)
    _, symmetric, trace, z = _davenport_matrix(body, reference, weights)
    davenport = np.zeros((4, 4))
    davenport[:3, :3] = symmetric - trace * np.eye(3)
    davenport[:3, 3] = z
    davenport[3, :3] = z
    davenport[3, 3] = trace
    eigenvalues, eigenvectors = np.linalg.eigh(davenport)
    optimum = eigenvectors[:, -1]
    attitude = _rotation_from_davenport_quaternion(optimum[:3], optimum[3])  # reference -> body
    attitude = _refine(attitude, body, reference, weights, refine_steps)
    quaternion = quaternion_from_rotation_matrix(attitude.T)
    return AttitudeSolution(
        quaternion=quaternion,
        covariance=wahba_covariance(body, sigmas),
        loss=_loss(attitude, body, reference, weights),
        method="q_method",
    )


def _quest_core(symmetric: np.ndarray, trace: float, z: np.ndarray, *, tolerance: float, max_iterations: int):
    """Return Davenport's [vector; scalar] quaternion, its scalar factor and lambda_max."""
    trace_adjugate = 0.5 * (np.trace(symmetric) ** 2 - np.trace(symmetric @ symmetric))
    determinant = float(np.linalg.det(symmetric))
    a = trace**2 - trace_adjugate
    b = trace**2 + z @ z
    c = determinant + z @ symmetric @ z
    d = z @ symmetric @ symmetric @ z
    constant = a * b + c * trace - d
    lam = 1.0  # weights are normalised, so the optimum is at most 1 and close to it
    for _ in range(max_iterations):
        value = lam**4 - (a + b) * lam**2 - c * lam + constant
        slope = 4.0 * lam**3 - 2.0 * (a + b) * lam - c
        if slope == 0.0:
            break
        update = value / slope
        lam -= update
        if abs(update) < tolerance:
            break
    alpha = lam**2 - trace**2 + trace_adjugate
    beta = lam - trace
    gamma = (lam + trace) * alpha - determinant
    x = (alpha * np.eye(3) + beta * symmetric + symmetric @ symmetric) @ z
    return x, gamma, lam


def quest(
    body: np.ndarray,
    reference: np.ndarray,
    sigma: np.ndarray | float,
    *,
    tolerance: float = 1.0e-12,
    max_iterations: int = 50,
    refine_steps: int = 8,
) -> AttitudeSolution:
    """QUEST (Shuster and Oh 1981): the Wahba optimum without an eigen-solver.

    The largest eigenvalue of Davenport's matrix is found by Newton iteration
    on its characteristic polynomial, starting from the sum of the weights,
    and the quaternion follows from the Gibbs vector. Near a 180 degree
    rotation the Gibbs vector is singular; the reference directions are then
    rotated by 180 degrees about a body axis (the "sequential rotations" of
    the original paper), the problem solved there, and the rotation undone.

    ``refine_steps`` Gauss-Newton steps (at most; they stop once a step is
    below 1e-14 rad) polish the closed-form answer. They matter when one
    direction is weighted far more heavily than the others, where the
    closed form loses accuracy (see :func:`_refine`); 0 gives the textbook
    algorithm.
    """
    body = _unit_rows(body, "body")
    reference = _unit_rows(reference, "reference")
    if body.shape != reference.shape:
        raise ValueError("body and reference must have the same shape")
    if body.shape[0] < 2:
        raise ValueError("at least two directions are needed to determine the attitude")
    sigmas = _sigmas(sigma, body.shape[0])
    weights = 1.0 / sigmas**2
    weights = weights / np.sum(weights)
    # Solve in four frames: the problem as posed, and with the reference
    # directions rotated 180 degrees about each axis. Near a 180 degree
    # attitude both parts of the unnormalised quaternion [x; gamma] vanish
    # together, so their magnitude, not the normalised scalar, tells which
    # frame is well conditioned; the largest wins.
    best = None
    rotations = [np.eye(3)] + [np.diag(signs) for signs in ((1.0, -1.0, -1.0), (-1.0, 1.0, -1.0), (-1.0, -1.0, 1.0))]
    for rotation in rotations:
        _, symmetric, trace, z = _davenport_matrix(body, reference @ rotation.T, weights)
        x, gamma, lam = _quest_core(symmetric, trace, z, tolerance=tolerance, max_iterations=max_iterations)
        magnitude = float(np.sqrt(gamma**2 + x @ x))
        if best is None or magnitude > best[0]:
            best = (magnitude, x / magnitude if magnitude > 0.0 else x, gamma / magnitude if magnitude > 0.0 else gamma, lam, rotation)
    _, vector, scalar, lam, rotation = best
    attitude_rotated = _rotation_from_davenport_quaternion(vector, scalar)  # rotated reference -> body
    attitude = attitude_rotated @ rotation  # reference -> body
    attitude = _refine(attitude, body, reference, weights, refine_steps)
    quaternion = quaternion_from_rotation_matrix(attitude.T)
    return AttitudeSolution(
        quaternion=quaternion,
        covariance=wahba_covariance(body, sigmas),
        loss=_loss(attitude, body, reference, weights),
        method="quest",
    )
