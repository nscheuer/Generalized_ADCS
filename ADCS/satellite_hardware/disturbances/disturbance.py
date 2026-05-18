__all__ = ["Disturbance"]

import numpy as np
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

        # Disturbance-parameter-estimation interface consumed by
        # EstimatedSatellite.match_estimate (restored: this was working in
        # the original PhD/thesis code and rotted in the port -- no
        # disturbance class implemented it, so estimating ANY disturbance
        # parameter raised AttributeError).
        #
        # `active`: a Drag-only deactivation flag; default True so every
        #   disturbance exposes it (a disturbance with estimate_dist=True is
        #   active for estimation by definition).
        # `std`: the estimated-parameter 1-sigma vector, length
        #   `estimated_vector_length`.
        # Estimable disturbances (`estimated_vector_length > 0`) MUST also
        # implement the `main_param` property (the estimated parameter
        # vector, read by the estimator and written back by match_estimate);
        # the base raises a clear error rather than silently mis-estimating.
        self.active = True
        self.std = np.zeros(int(estimated_vector_length))

    @property
    def main_param(self) -> "np.ndarray":
        raise NotImplementedError(
            f"{type(self).__name__} declares estimated_vector_length="
            f"{getattr(self, 'estimated_vector_length', 0)} but does not "
            f"implement the `main_param` parameter vector required for "
            f"disturbance-parameter estimation."
        )

    @main_param.setter
    def main_param(self, value) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement a settable "
            f"`main_param`; cannot write back an estimated disturbance "
            f"parameter."
        )
    