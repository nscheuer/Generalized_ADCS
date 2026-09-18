"""
Goal Formulation Stage — full 7-step pipeline.

Translates a GoalSpec + spacecraft state into standardized error
signals for any control law, handling all 4 cells of the
(goal_type x law_attitude_type) conversion table.

Also provides the Phase 1 legacy entry point for backward compatibility.
"""

__all__ = ["goal_formulation_step", "goal_formulation_step_legacy"]

import numpy as np
import warnings

from ADCS.pipeline.data import (
    GoalSpec,
    GoalFormulationOutput,
    LawInterface,
)
from ADCS.helpers.math_helpers import rot_mat, normalize
from ADCS.pipeline.goal_formulation.normalize_goal import (
    normalize_full_goal,
    normalize_reduced_goal,
)
from ADCS.pipeline.goal_formulation.world_vectors import resolve_world_vector
from ADCS.pipeline.goal_formulation.omega_ref import compute_omega_ref_world
from ADCS.pipeline.goal_formulation.attitude_error import (
    attitude_full_to_full,
    attitude_reduced_to_full,
    attitude_reduced_to_reduced,
    attitude_full_to_reduced,
    attitude_none,
    AlternatingState,
)
from ADCS.orbits.orbital_state import Orbital_State


def goal_formulation_step(
    goal_spec: GoalSpec,
    q: np.ndarray,
    omega: np.ndarray,
    os: Orbital_State,
    law_flags: LawInterface,
    goal_spec_next: GoalSpec = None,
    persistent_state: AlternatingState = None,
    dt: float = 1.0,
    epsilon_reg: float = 1e-6,
    alternating_body_vectors: tuple = None,
    alternating_switch: str = 'every_step',
    alternating_threshold: float = 0.01,
    alternating_period: int = 10,
) -> GoalFormulationOutput:
    """Full 7-step goal formulation pipeline.

    Parameters
    ----------
    goal_spec : GoalSpec
        Current goal specification.
    q : ndarray, shape (4,)
        Current attitude quaternion (Hamilton, scalar-first, body-to-ECI).
    omega : ndarray, shape (3,)
        Current body angular velocity (rad/s).
    os : Orbital_State
        Current orbital state.
    law_flags : LawInterface
        Control law declarations.
    goal_spec_next : GoalSpec or None
        Next-step goal for finite differencing.
    persistent_state : AlternatingState or None
        Persistent state for full->reduced alternating.
    dt : float
        Control timestep for finite differencing.
    epsilon_reg : float
        Anti-parallel regularization strength.
    alternating_body_vectors : tuple or None
        (b1, b2) custom body vectors for alternating decomposition.
    alternating_switch : str
        Alternating strategy: 'every_step', 'threshold', 'time_based'.
    alternating_threshold : float
        Error threshold for threshold-based switching (rad).
    alternating_period : int
        Period for time-based switching (timesteps).

    Returns
    -------
    GoalFormulationOutput
        Standardized error signals for the control law and compensation.
    """
    # ==================================================================
    # STEP 1: NORMALIZE GOAL
    # ==================================================================
    goal_type = goal_spec.goal_type
    q_g = None
    b_hat = None
    u_hat = None

    if goal_type == 'full':
        q_g = normalize_full_goal(goal_spec)

    elif goal_type == 'reduced':
        # Resolve world vector
        if goal_spec.u_hat_eci is not None:
            u_hat_resolved = normalize(goal_spec.u_hat_eci)
        elif goal_spec.u_spec is not None:
            u_hat_resolved = resolve_world_vector(goal_spec.u_spec, os)
        else:
            raise ValueError("Reduced goal must have u_hat_eci or u_spec.")
        b_hat, u_hat = normalize_reduced_goal(goal_spec, u_hat_resolved)

    elif goal_type == 'none':
        pass
    else:
        raise ValueError(f"Unknown goal_type: {goal_type}")

    # Compute projection matrix
    if goal_type == 'reduced' and b_hat is not None:
        P = np.eye(3) - np.outer(b_hat, b_hat)
    elif goal_type == 'full':
        P = np.eye(3)
    else:  # none
        P = np.zeros((3, 3))

    # ==================================================================
    # STEP 2: COMPUTE OMEGA_REF (WORLD FRAME)
    # ==================================================================
    omega_ref_world = compute_omega_ref_world(
        goal_spec, goal_spec_next, goal_type, q, os, dt,
    )

    # ==================================================================
    # STEP 3: FRAME CONVERSION (world -> body)
    # ==================================================================
    R_b2i = rot_mat(q)
    omega_ref_body = R_b2i.T @ omega_ref_world

    # ==================================================================
    # STEP 4: COMPUTE OMEGA ERROR
    # ==================================================================
    omega_raw_error = omega - omega_ref_body
    omega_e = P @ omega_raw_error

    # ==================================================================
    # STEP 5: COMPUTE ATTITUDE ERROR (2x2 conversion table)
    # ==================================================================
    P_final = P

    if goal_type == 'none':
        attitude_output = attitude_none(law_flags)

    elif goal_type == 'full' and law_flags.attitude_type == 'full':
        attitude_output = attitude_full_to_full(q_g, q, law_flags)

    elif goal_type == 'reduced' and law_flags.attitude_type == 'full':
        attitude_output = attitude_reduced_to_full(
            b_hat, u_hat, q, law_flags, epsilon_reg,
        )

    elif goal_type == 'reduced' and law_flags.attitude_type == 'reduced':
        attitude_output = attitude_reduced_to_reduced(
            b_hat, u_hat, q, law_flags,
        )

    elif goal_type == 'full' and law_flags.attitude_type == 'reduced':
        if persistent_state is None:
            persistent_state = AlternatingState()
        attitude_output, P_final, persistent_state = attitude_full_to_reduced(
            q_g, q, law_flags, persistent_state,
            alternating_body_vectors, alternating_switch,
            alternating_threshold, alternating_period,
        )
        # Recompute omega_e with updated P from active sub-goal
        omega_e = P_final @ omega_raw_error

    else:
        raise ValueError(
            f"Unhandled goal_type={goal_type}, "
            f"law attitude_type={law_flags.attitude_type}"
        )

    # ==================================================================
    # STEP 6: APPLY OMEGA FLAGS
    # ==================================================================
    if law_flags.omega_type == 'omega_error':
        omega_output = omega_e
        inject_damping = False
    elif law_flags.omega_type == 'omega_raw':
        omega_output = omega_raw_error
        inject_damping = False
    else:  # 'no_omega'
        omega_output = None
        inject_damping = True

    # ==================================================================
    # STEP 7: CONVENTION CONVERSION
    # ==================================================================
    # Already applied inside attitude_full_to_full and attitude_reduced_to_full.
    # For reduced law outputs, no quaternion convention needed.

    return GoalFormulationOutput(
        attitude_output=attitude_output,
        omega_output=omega_output,
        P=P_final,
        omega_ref_body=omega_ref_body,
        goal_type=goal_type,
        inject_damping=inject_damping,
    )


# ======================================================================
# Phase 1 legacy entry point (backward compatible)
# ======================================================================

def goal_formulation_step_legacy(
    q_err: np.ndarray,
    omega: np.ndarray,
    omega_ref_eci: np.ndarray,
    q: np.ndarray,
    law_interface: LawInterface,
) -> GoalFormulationOutput:
    """Phase 1 legacy entry point using pre-computed Goal.error() outputs.

    This is the original Phase 1 implementation that works with the
    existing Goal.error() / Goal.to_ref() interface. PipelineController
    uses this for backward compatibility with legacy Goal objects.

    Parameters
    ----------
    q_err : ndarray, shape (3,)
        Attitude error vector from Goal.error().
    omega : ndarray, shape (3,)
        Current body angular velocity.
    omega_ref_eci : ndarray, shape (3,)
        Reference angular velocity in ECI frame.
    q : ndarray, shape (4,)
        Current attitude quaternion (Hamilton, scalar-first).
    law_interface : LawInterface
        Declares what the control law expects.

    Returns
    -------
    GoalFormulationOutput
        Standardized error signals.
    """
    R_b2i = rot_mat(q)
    omega_ref_body = R_b2i.T @ omega_ref_eci
    P = np.eye(3)

    if law_interface.omega_type == 'omega_error':
        omega_output = omega - omega_ref_body
    elif law_interface.omega_type == 'omega_raw':
        omega_output = omega.copy()
    else:
        omega_output = None

    inject_damping = (law_interface.omega_type == 'no_omega')

    return GoalFormulationOutput(
        attitude_output=q_err,
        omega_output=omega_output,
        P=P,
        omega_ref_body=omega_ref_body,
        goal_type='full',
        inject_damping=inject_damping,
    )
