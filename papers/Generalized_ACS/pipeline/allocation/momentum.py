"""
Momentum management / wheel desaturation strategies.

Three strategies are supported:

1. **Nullspace** (overactuated systems, e.g. 3RW+3MTQ):
   Project a desaturation command into the nullspace of B_tau so
   it has zero impact on the achieved torque.

2. **Weighted** (general, trades pointing for momentum management):
   Augment the QP cost with a desaturation term.  The optimizer
   balances pointing vs. desaturation via w_desat.

3. **Scheduled** (underactuated, e.g. 3MTQ+1RW):
   Only desaturate during orbit phases with favorable MTQ authority.
   Adds the desat torque directly to tau_desired when activated.

All strategies compute a desired dump torque from the RW momentum error:
    tau_desat = -k_desat * (h_rw_body - h_target)
"""

__all__ = [
    "apply_nullspace_desaturation",
    "build_weighted_desat_system",
    "apply_scheduled_desaturation",
    "compute_desat_torque",
    "compute_mtq_authority",
]

import numpy as np
from typing import List, Optional, Tuple

from ADCS.pipeline.data import ActuatorGroup, DesaturationConfig


def compute_desat_torque(
    h_rw_body: np.ndarray,
    desat_config: DesaturationConfig,
) -> np.ndarray:
    """Compute desired desaturation torque from RW momentum error.

    Parameters
    ----------
    h_rw_body : ndarray, shape (3,)
        Total RW angular momentum in body frame.
    desat_config : DesaturationConfig
        Desaturation parameters.

    Returns
    -------
    tau_desat : ndarray, shape (3,)
        Desired desaturation torque in body frame.
    """
    h_err = h_rw_body - desat_config.h_rw_target
    if np.linalg.norm(h_err) < desat_config.h_rw_threshold:
        return np.zeros(3)
    return -desat_config.k_desat * h_err


def apply_nullspace_desaturation(
    u_primary: np.ndarray,
    B_tau: np.ndarray,
    u_min: np.ndarray,
    u_max: np.ndarray,
    h_rw_body: np.ndarray,
    rw_columns: np.ndarray,
    desat_config: DesaturationConfig,
) -> np.ndarray:
    """Apply nullspace desaturation to an existing primary solution.

    Projects the desired desaturation command into the nullspace of B_tau
    so there is zero torque impact on the primary control objective.

    Parameters
    ----------
    u_primary : ndarray, shape (n,)
        Primary allocation solution (from LP/QP/etc).
    B_tau : ndarray, shape (3, n)
        Torque effectiveness matrix.
    u_min, u_max : ndarray, shape (n,)
        Actuator command bounds.
    h_rw_body : ndarray, shape (3,)
        Total RW angular momentum in body frame.
    rw_columns : ndarray of int
        Column indices in B_tau corresponding to RW actuators.
    desat_config : DesaturationConfig
        Desaturation parameters.

    Returns
    -------
    u_combined : ndarray, shape (n,)
        Primary + nullspace desaturation commands, clipped to bounds.
    """
    n = B_tau.shape[1]
    tau_desat = compute_desat_torque(h_rw_body, desat_config)

    if np.linalg.norm(tau_desat) < 1e-12:
        return u_primary.copy()

    # Compute nullspace of B_tau via SVD
    U, S, Vt = np.linalg.svd(B_tau, full_matrices=True)
    rank = np.sum(S > 1e-10)

    if rank >= n:
        # No nullspace available (square or underdetermined)
        return u_primary.copy()

    N = Vt[rank:, :].T  # [n x (n - rank)]

    # Build desired desaturation command vector (only RW axes)
    # tau_desat = A_rw @ u_desat_rw  =>  u_desat_rw = pinv(A_rw) @ tau_desat
    # We want the full n-vector with desat commands only at RW columns
    u_desat_desired = np.zeros(n)
    if len(rw_columns) > 0:
        A_rw = B_tau[:, rw_columns]
        u_desat_rw = np.linalg.pinv(A_rw) @ tau_desat
        u_desat_desired[rw_columns] = u_desat_rw

    # Project into nullspace
    u_desat_null = N @ (N.T @ u_desat_desired)

    # Coupled scaling to respect bounds (like the LP controller pattern)
    u_combined = u_primary + u_desat_null
    beta = _compute_coupled_scale(u_primary, u_desat_null, u_min, u_max)
    u_combined = u_primary + beta * u_desat_null

    return u_combined


def build_weighted_desat_system(
    B_tau: np.ndarray,
    tau_desired: np.ndarray,
    u_min: np.ndarray,
    u_max: np.ndarray,
    h_rw_body: np.ndarray,
    actuator_groups: list,
    desat_config: DesaturationConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build augmented system for weighted desaturation QP.

    Augments the torque-tracking cost with a desaturation term:
        min ||B_tau @ u - tau_desired||^2 + w_desat * ||A_desat @ u - tau_desat||^2

    This is reformulated as a single least-squares problem:
        min ||A_aug @ u - b_aug||^2   s.t.  u_min <= u <= u_max

    Parameters
    ----------
    B_tau : ndarray, shape (3, n)
        Torque effectiveness matrix.
    tau_desired : ndarray, shape (3,)
        Desired control torque.
    u_min, u_max : ndarray, shape (n,)
        Actuator command bounds.
    h_rw_body : ndarray, shape (3,)
        Total RW angular momentum in body frame.
    actuator_groups : list of ActuatorGroup
        For identifying RW columns.
    desat_config : DesaturationConfig
        Desaturation parameters.

    Returns
    -------
    A_aug : ndarray, shape (6, n)
        Augmented system matrix [B_tau; sqrt(w) * A_desat].
    b_aug : ndarray, shape (6,)
        Augmented RHS [tau_desired; sqrt(w) * tau_desat].
    """
    n = B_tau.shape[1]
    tau_desat = compute_desat_torque(h_rw_body, desat_config)

    # Build A_desat: maps actuator commands to desaturation torque
    # Only RW columns contribute to momentum change
    A_desat = np.zeros((3, n))
    col_offset = 0
    for group in actuator_groups:
        n_g = group.axes.shape[1]
        if group.group_type == 'rw':
            A_desat[:, col_offset:col_offset + n_g] = group.axes
        col_offset += n_g

    w_sqrt = np.sqrt(desat_config.w_desat)

    A_aug = np.vstack([B_tau, w_sqrt * A_desat])
    b_aug = np.concatenate([tau_desired, w_sqrt * tau_desat])

    return A_aug, b_aug


def compute_mtq_authority(
    B_body: np.ndarray,
    actuator_groups: list,
) -> float:
    """Compute the current MTQ torque authority as a fraction of maximum.

    Authority is measured as the ratio of the effective MTQ torque
    magnitude to its maximum possible value. This depends on the
    alignment of B_body with the MTQ axes — when B is parallel to
    an MTQ axis, that axis produces zero torque.

    Parameters
    ----------
    B_body : ndarray, shape (3,)
        Magnetic field in body frame.
    actuator_groups : list of ActuatorGroup
        Actuator groups.

    Returns
    -------
    authority : float
        Fraction in [0, 1]. 1.0 = full authority, 0.0 = no MTQ torque possible.
    """
    from ADCS.helpers.math_helpers import skewsym

    mtq_group = None
    for group in actuator_groups:
        if group.group_type == 'mtq':
            mtq_group = group
            break

    if mtq_group is None:
        return 0.0

    B_norm = np.linalg.norm(B_body)
    if B_norm < 1e-12:
        return 0.0

    # Effective torque matrix for MTQs
    B_skew = skewsym(B_body)
    M_eff = -B_skew @ mtq_group.axes  # [3 x n_mtq]

    # Maximum torque using all MTQs at u_max
    # The effective torque depends on the angle between B and each MTQ axis
    # Use the matrix singular values as a measure of achievable torque
    S = np.linalg.svd(M_eff, compute_uv=False)

    # Maximum effectiveness: if B were perpendicular to all axes
    # M_max = B_norm * A_mtq (maximum cross product)
    S_max = np.linalg.svd(B_norm * mtq_group.axes, compute_uv=False)

    if np.max(S_max) < 1e-12:
        return 0.0

    # Authority = ratio of actual to maximum singular value spread
    # Use the minimum singular value ratio as the bottleneck measure
    authority = np.min(S) / np.max(S_max)
    return float(np.clip(authority, 0.0, 1.0))


def apply_scheduled_desaturation(
    tau_desired: np.ndarray,
    h_rw_body: np.ndarray,
    B_body: np.ndarray,
    actuator_groups: list,
    desat_config: DesaturationConfig,
) -> np.ndarray:
    """Apply scheduled desaturation by modifying tau_desired.

    Only adds desaturation torque when MTQ authority is above threshold,
    allowing the allocator to distribute the combined torque naturally.

    Parameters
    ----------
    tau_desired : ndarray, shape (3,)
        Primary desired torque.
    h_rw_body : ndarray, shape (3,)
        Total RW angular momentum in body frame.
    B_body : ndarray, shape (3,)
        Magnetic field in body frame.
    actuator_groups : list of ActuatorGroup
        Actuator groups.
    desat_config : DesaturationConfig
        Desaturation parameters.

    Returns
    -------
    tau_combined : ndarray, shape (3,)
        tau_desired + tau_desat (if conditions met), otherwise tau_desired.
    """
    tau_desat = compute_desat_torque(h_rw_body, desat_config)

    if np.linalg.norm(tau_desat) < 1e-12:
        return tau_desired.copy()

    authority = compute_mtq_authority(B_body, actuator_groups)

    if authority < desat_config.authority_threshold:
        return tau_desired.copy()

    return tau_desired + tau_desat


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_coupled_scale(
    u_primary: np.ndarray,
    u_secondary: np.ndarray,
    u_min: np.ndarray,
    u_max: np.ndarray,
) -> float:
    """Compute maximum scalar beta such that u_primary + beta * u_secondary
    stays within [u_min, u_max].

    Uses the LP controller's coupled-scaling pattern: a single scalar
    preserves the torque-free balance of the secondary command.
    """
    beta = 1.0
    for i in range(len(u_secondary)):
        si = u_secondary[i]
        if abs(si) < 1e-15:
            continue
        if si > 0:
            margin = u_max[i] - u_primary[i]
        else:
            margin = u_primary[i] - u_min[i]
        beta = min(beta, margin / abs(si))

    return float(np.clip(beta, 0.0, 1.0))
