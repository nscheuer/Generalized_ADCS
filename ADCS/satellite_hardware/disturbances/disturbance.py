__all__ = ["Disturbance"]

from typing import List

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
    