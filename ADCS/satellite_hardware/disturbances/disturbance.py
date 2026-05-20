__all__ = ["Disturbance"]

import numpy as np
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ADCS.orbits.orbital_state import Orbital_State

class Disturbance:
    def __init__(self, estimate_dist: bool = False, estimated_vector_length: int = 0):
        r"""
        Base Class for Disturbance Models.

        This class defines the **common interface and configuration** for all
        disturbance models used in the spacecraft dynamics and attitude determination
        and control system (ADCS) framework.

        A disturbance represents any **non-commanded force or torque** acting on the
        spacecraft, such as magnetic dipole torques, aerodynamic drag, gravity
        gradient effects, or solar radiation pressure.

        Disturbance Estimation Concept
        ------------------------------
        Disturbances may optionally be treated as **estimable parameters** within
        a state estimation framework (e.g., EKF, UKF).

        Let the spacecraft dynamics be written as

        .. math::

            \dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}, \mathbf{u}) + \mathbf{d},

        where :math:`\mathbf{d}` represents a disturbance contribution.

        When disturbance estimation is enabled, the disturbance vector is augmented
        into the estimator state:

        .. math::

            \mathbf{x}_\text{aug}
            =
            \begin{bmatrix}
                \mathbf{x} \\
                \mathbf{d}
            \end{bmatrix}.

        The length of the disturbance subvector is specified by
        ``estimated_vector_length``.

        Design Intent
        -------------
        This class is intended to be **subclassed**, not instantiated directly.
        Derived classes implement the physical disturbance model and optionally
        provide Jacobians and Hessians for use in estimation or optimization.

        :param estimate_dist: Enables augmentation of the disturbance into the
            estimator state vector.
        :type estimate_dist: bool
        :param estimated_vector_length: Length of the disturbance parameter vector
            to be estimated.
        :type estimated_vector_length: int
        :return: None
        :rtype: None
        """
        self.estimate_dist = estimate_dist
        self.estimated_vector_length = estimated_vector_length

    # ------------------------------------------------------------------
    # Canonical analytic-derivative interface.
    #
    # Every consumer (`Satellite.dist_torques_jacobian`, `dist_torque_hess`,
    # the dynamics-Hessians chain, an eventual EKF / one-step MPC) calls
    # `j.torque_qjac` / `torque_qqhess` / `torque_valjac` /
    # `torque_qvalhess` / `torque_valvalhess` on EVERY disturbance, not just
    # the estimable ones. Without a default, any subclass that didn't
    # implement a particular derivative raised AttributeError, making the
    # whole chain dead-on-arrival for that disturbance combination.
    #
    # The base provides correctly-shaped ZERO defaults; subclasses override
    # with real physics. This is the standard pattern used by base
    # `Actuator` (zero-impls for every `dtorq__*` / `ddtorq__*` /
    # `ddstor_torq__*`) and base `Sensor` (defaults for `reading` /
    # `basestate_jac`). All derivatives use the unified signature
    # ``(self, sat, x, os)`` -- consistent with Drag / GG / SRP, and the
    # Dipole / Prop subclasses are updated to match.
    # ------------------------------------------------------------------

    def torque_qjac(self, sat, x: np.ndarray, os: "Orbital_State") -> np.ndarray:
        r"""Quaternion Jacobian of the disturbance torque. Default = zeros (4, 3)."""
        return np.zeros((4, 3))

    def torque_qqhess(self, sat, x: np.ndarray, os: "Orbital_State") -> np.ndarray:
        r"""Quaternion Hessian of the disturbance torque. Default = zeros (4, 4, 3)."""
        return np.zeros((4, 4, 3))

    def torque_valjac(self, sat, x: np.ndarray, os: "Orbital_State") -> np.ndarray:
        r"""Disturbance-parameter Jacobian. Default = zeros (estimated_vector_length, 3)."""
        return np.zeros((int(self.estimated_vector_length), 3))

    def torque_qvalhess(self, sat, x: np.ndarray, os: "Orbital_State") -> np.ndarray:
        r"""Mixed quaternion / disturbance-parameter Hessian. Default = zeros (4, val, 3)."""
        return np.zeros((4, int(self.estimated_vector_length), 3))

    def torque_valvalhess(self, sat, x: np.ndarray, os: "Orbital_State") -> np.ndarray:
        r"""Disturbance-parameter Hessian. Default = zeros (val, val, 3)."""
        return np.zeros((int(self.estimated_vector_length), int(self.estimated_vector_length), 3))
    