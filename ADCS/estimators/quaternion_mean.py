"""Chart-consistent quaternion means for unscented attitude filters."""

import numpy as np

from ADCS.helpers.math_helpers import quat_mult
from ADCS.state import State


def _skew(vectors: np.ndarray) -> np.ndarray:
    matrices = np.zeros((len(vectors), 3, 3))
    x, y, z = vectors.T
    matrices[:, 0, 1], matrices[:, 1, 0] = -z, z
    matrices[:, 0, 2], matrices[:, 2, 0] = y, -y
    matrices[:, 1, 2], matrices[:, 2, 1] = -x, x
    return matrices


def _deviations(
    quaternion: np.ndarray, values: np.ndarray, mode: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return shortest right errors and their rotation-vector derivatives."""
    scalar, vector = quaternion[0], quaternion[1:]
    relative_scalar = scalar * values[:, 0] + values[:, 1:] @ vector
    relative_vector = (
        scalar * values[:, 1:]
        - values[:, 0, None] * vector
        - np.cross(vector, values[:, 1:])
    )
    signs = np.where(relative_scalar < 0.0, -1.0, 1.0)
    s = relative_scalar * signs
    v = relative_vector * signs[:, None]
    identity = np.eye(3)
    # q' = Exp(-step) q: ds/dstep = v/2 and dv/dstep = (-s I + [v]x)/2.
    vector_derivative = 0.5 * (-s[:, None, None] * identity + _skew(v))

    if mode == "quaternion_vector":
        return 2.0 * v, 2.0 * vector_derivative
    if mode == "rotation_vector":
        length = np.linalg.norm(v, axis=1)
        angle = 2.0 * np.arctan2(length, s)
        scale = np.full_like(length, 2.0)
        np.divide(angle, length, out=scale, where=length > 1.0e-15)
        errors = scale[:, None] * v
        coefficient = np.full_like(angle, 1.0 / 12.0)
        large = angle > 1.0e-4
        theta = angle[large]
        coefficient[large] = (1.0 - 0.5 * theta / np.tan(0.5 * theta)) / theta**2
        skew = _skew(errors)
        derivative = -identity + 0.5 * skew - coefficient[:, None, None] * (skew @ skew)
        return errors, derivative

    divisor = s if mode == "cayley" else 1.0 + s
    if np.any(divisor < 1.0e-10):
        raise ValueError(
            "UKF attitude mean reaches the Cayley chart singularity at 180 degrees; "
            "use rotation_vector coordinates"
        )
    scale = 2.0 if mode == "two_mrp" else 1.0
    errors = scale * v / divisor[:, None]
    derivative = scale * (
        vector_derivative / divisor[:, None, None]
        - 0.5 * v[:, :, None] * v[:, None, :] / divisor[:, None, None] ** 2
    )
    return errors, derivative


def quaternion_mean(values: np.ndarray, weights: np.ndarray, *, mode: str) -> np.ndarray:
    """Solve sum(w_i * chart(q^-1 q_i)) = 0, including signed UKF weights.

    Newton steps use rotation vectors, so the solver never retracts an
    out-of-domain quaternion-vector correction. Backtracking limits each step
    and reduces the residual, unlike the undamped chart fixed point. Both
    state and measurement means use this same definition; covariance errors
    remain in the requested chart. The analytic 3x3 Jacobian avoids repeated
    state copies and numerical differentiation.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    norms = np.linalg.norm(values, axis=1)
    if not np.all(np.isfinite(values)) or np.any(norms == 0.0):
        raise ValueError("UKF quaternion mean requires finite nonzero quaternions")
    values = values / norms[:, None]
    mean = values[0].copy()
    for _ in range(32):
        deviations, derivatives = _deviations(mean, values, mode)
        residual = weights @ deviations
        residual_norm = np.linalg.norm(residual)
        # Allow for roundoff in cancellation between signed sigma weights.
        scale = np.abs(weights) @ np.linalg.norm(deviations, axis=1)
        tolerance = 1.0e-12 + 64.0 * np.finfo(float).eps * scale
        if residual_norm <= tolerance:
            return mean
        jacobian = np.einsum("i,ijk->jk", weights, derivatives)
        try:
            step = np.linalg.solve(jacobian, -residual)
        except np.linalg.LinAlgError:
            break
        step_norm = np.linalg.norm(step)
        if not np.isfinite(step_norm):
            break
        step *= min(1.0, (np.pi / 2.0) / max(step_norm, 1.0e-15))
        for _ in range(16):
            candidate = quat_mult(
                mean, State.quaternion_delta_from_vector(step, mode="rotation_vector")
            )
            candidate /= np.linalg.norm(candidate)
            try:
                candidate_deviations, _ = _deviations(candidate, values, mode)
                improved = np.linalg.norm(weights @ candidate_deviations) < residual_norm
            except ValueError:
                improved = False
            if improved:
                mean = candidate
                break
            step *= 0.5
        else:
            break
    raise RuntimeError(
        f"UKF quaternion mean did not converge in {mode!r} coordinates; "
        "the prior may be too broad for a single local attitude distribution"
    )
