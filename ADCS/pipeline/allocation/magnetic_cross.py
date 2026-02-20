"""
Magnetic cross-product torque allocation.

Computes the magnetic dipole command that best approximates the
desired torque using only magnetorquers:

    m = (B x tau_des) / |B|^2

This projects the desired torque onto the plane perpendicular to B,
which is the achievable torque subspace for magnetorquers.

Direction-preserving saturation scales the entire dipole vector
uniformly to respect actuator limits, preserving the torque direction.
"""

__all__ = ["allocate_magnetic_cross"]

import numpy as np

from ADCS.pipeline.data import AllocationResult, ActuatorGroup


def allocate_magnetic_cross(
    tau_desired: np.ndarray,
    B_body: np.ndarray,
    mtq_group: ActuatorGroup,
    n_actuators: int,
) -> AllocationResult:
    """Allocate torque to magnetorquers via cross-product inversion.

    Parameters
    ----------
    tau_desired : ndarray, shape (3,)
        Desired torque in body frame.
    B_body : ndarray, shape (3,)
        Magnetic field vector in body frame (Tesla).
    mtq_group : ActuatorGroup
        Magnetorquer actuator group with axes and limits.
    n_actuators : int
        Total number of actuators in the spacecraft.

    Returns
    -------
    AllocationResult
        Actuator command vector and allocation metadata.
    """
    B_norm_sq = np.dot(B_body, B_body)

    if B_norm_sq < 1e-11:
        u_out = np.zeros(n_actuators)
        return AllocationResult(u=u_out, tau_achieved=np.zeros(3), alpha=0.0)

    # Cross-product allocation: m = (B x tau) / |B|^2
    m_cmd = np.cross(B_body, tau_desired) / B_norm_sq

    # Direction-preserving saturation: scale uniformly
    u_max = mtq_group.u_max
    ratios = np.where(
        np.abs(m_cmd) > 0.0,
        u_max / np.abs(m_cmd),
        np.inf,
    )
    alpha = min(1.0, float(np.min(ratios)))
    m_cmd *= alpha

    # Achieved torque: tau = m x B
    tau_achieved = np.cross(m_cmd, B_body)

    # Pack into full actuator command vector
    u_out = np.zeros(n_actuators)
    if mtq_group.indices is not None:
        u_out[mtq_group.indices] = m_cmd
    else:
        # Assume MTQs are contiguous; caller should set indices
        u_out[:len(m_cmd)] = m_cmd

    return AllocationResult(
        u=u_out,
        tau_achieved=tau_achieved,
        alpha=alpha,
    )
