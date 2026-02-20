"""
Compensation stage: adds feedforward and correction terms to the
control law output.

Phase 1: only gyroscopic compensation is implemented.
Future phases add frame rotation feedforward, disturbance feedforward,
and damping injection.
"""

__all__ = ["compensation_step"]

import numpy as np

from ADCS.pipeline.data import CompensationConfig, CompensationInputs
from ADCS.pipeline.compensation.gyroscopic import compute_gyroscopic_torque


def compensation_step(
    tau_law: np.ndarray,
    omega: np.ndarray,
    J: np.ndarray,
    h_rw_body: np.ndarray,
    comp_config: CompensationConfig,
    comp_inputs: CompensationInputs,
) -> np.ndarray:
    """Apply compensation terms to the control law torque.

    The total desired torque is:
        tau_desired = tau_law + sum(enabled compensation terms)

    Parameters
    ----------
    tau_law : ndarray, shape (3,)
        Torque output from the control law.
    omega : ndarray, shape (3,)
        Body angular velocity.
    J : ndarray, shape (3, 3)
        Spacecraft inertia matrix.
    h_rw_body : ndarray, shape (3,)
        Total reaction wheel angular momentum in body frame.
    comp_config : CompensationConfig
        Toggle flags for compensation terms.
    comp_inputs : CompensationInputs
        Data from goal formulation (P, omega_ref_body, etc.).

    Returns
    -------
    ndarray, shape (3,)
        Total desired torque after compensation.
    """
    tau_total = tau_law.copy()

    if comp_config.enable_gyroscopic:
        tau_total += compute_gyroscopic_torque(omega, J, h_rw_body)

    # Phase 2+: frame rotation feedforward
    # Phase 2+: disturbance feedforward
    # Phase 2+: damping injection

    return tau_total
