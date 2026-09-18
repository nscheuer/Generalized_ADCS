"""
Compensation stage: adds feedforward and correction terms to the
control law output.

Supported compensation terms:
    - Gyroscopic:       omega x (J @ omega + h_rw_body)
    - Frame rotation:   J @ omega_ref_dot - cross(omega_ref, J @ omega_ref)
    - Disturbance FF:   -(tau_gg + tau_mag + ...)
    - Damping injection: -k_d * P @ (omega - omega_ref)
"""

__all__ = ["compensation_step"]

import numpy as np

from ADCS.pipeline.data import CompensationConfig, CompensationInputs
from ADCS.pipeline.compensation.gyroscopic import compute_gyroscopic_torque
from ADCS.pipeline.compensation.frame_rotation import compute_frame_rotation_torque
from ADCS.pipeline.compensation.damping_injection import compute_damping_injection
from ADCS.pipeline.compensation.disturbance_ff import compute_disturbance_feedforward


def compensation_step(
    tau_law: np.ndarray,
    omega: np.ndarray,
    J: np.ndarray,
    h_rw_body: np.ndarray,
    comp_config: CompensationConfig,
    comp_inputs: CompensationInputs,
    omega_ref_body_prev: np.ndarray = None,
    dt: float = 1.0,
    q: np.ndarray = None,
    r_eci: np.ndarray = None,
    B_body: np.ndarray = None,
    m_residual: np.ndarray = None,
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
    omega_ref_body_prev : ndarray, shape (3,) or None
        Previous-step omega_ref_body for frame rotation finite diff.
    dt : float
        Timestep for finite differencing (s).
    q : ndarray, shape (4,) or None
        Attitude quaternion (needed for disturbance FF).
    r_eci : ndarray, shape (3,) or None
        Spacecraft position in ECI (needed for gravity gradient FF).
    B_body : ndarray, shape (3,) or None
        Magnetic field in body frame (needed for magnetic disturbance FF).
    m_residual : ndarray, shape (3,) or None
        Residual magnetic dipole (needed for magnetic disturbance FF).

    Returns
    -------
    ndarray, shape (3,)
        Total desired torque after compensation.
    """
    tau_total = tau_law.copy()

    if comp_config.enable_gyroscopic:
        tau_total += compute_gyroscopic_torque(omega, J, h_rw_body)

    if comp_config.enable_frame_rotation and omega_ref_body_prev is not None:
        tau_total += compute_frame_rotation_torque(
            comp_inputs.omega_ref_body,
            omega_ref_body_prev,
            J, dt,
        )

    if comp_config.enable_disturbance_ff and q is not None and r_eci is not None:
        tau_total += compute_disturbance_feedforward(
            q=q, r_eci=r_eci, J=J,
            B_body=B_body, m_residual=m_residual,
            enable_gravity_gradient=True,
            enable_magnetic=(B_body is not None and m_residual is not None),
        )

    if comp_config.enable_damping_injection and comp_inputs.inject_damping:
        tau_total += compute_damping_injection(
            omega=omega,
            omega_ref_body=comp_inputs.omega_ref_body,
            P=comp_inputs.P,
            k_d=comp_config.damping_gain,
        )

    return tau_total
