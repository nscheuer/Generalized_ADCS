__all__ = ["Disturbance"]

import numpy as np
from typing import List

from ADCS.covariance import Covariance
from ADCS.orbits.universal_constants import TimeConstants

class Disturbance:
    def __init__(
        self,
        estimate_dist: bool = False,
        estimated_vector_length: int = 0,
        parameter_std_rate: np.ndarray | float = 0.0,
    ):
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
        :param parameter_std_rate: Continuous random-walk standard-deviation
            rate for each estimated parameter.
        :type parameter_std_rate: float or numpy.ndarray
        :return: None
        :rtype: None
        """
        self.estimate_dist = estimate_dist
        self.estimated_vector_length = estimated_vector_length

        # Disturbance-parameter-estimation interface consumed by
        # EstimatedSatellite.match_estimate.
        #
        # `active`: a Drag-only deactivation flag; default True so every
        #   disturbance exposes it (a disturbance with estimate_dist=True is
        #   active for estimation by definition).
        # `std`: posterior estimated-parameter 1-sigma values.
        # `parameter_std_rate`: configured random-walk diffusion rates.  These
        #   must not be overwritten when an estimate is matched.
        # Estimable disturbances (`estimated_vector_length > 0`) MUST also
        # implement the `main_param` property (the estimated parameter
        # vector, read by the estimator and written back by match_estimate);
        # the base raises a clear error rather than silently mis-estimating.
        self.active = True
        self.std = np.zeros(int(estimated_vector_length))
        rate = np.asarray(parameter_std_rate, dtype=float)
        try:
            self.parameter_std_rate = np.broadcast_to(
                rate, (int(estimated_vector_length),)
            ).copy()
        except ValueError as error:
            raise ValueError(
                "parameter_std_rate must be scalar or match estimated_vector_length"
            ) from error
        if np.any(~np.isfinite(self.parameter_std_rate)) or np.any(
            self.parameter_std_rate < 0.0
        ):
            raise ValueError("parameter_std_rate must be finite and non-negative")
        self.last_update_time = float("nan")

    def update(self) -> None:
        """Refresh the values held over the next plant step.

        Disturbances with a noise model draw a fresh sample here; the base
        class has nothing to refresh. Called by :meth:`update_errors`.
        """

    def update_errors(self, j2000: float) -> None:
        r"""Advance the stochastic error models by one plant step.

        The estimated parameters (``main_param``) take one step of their
        random walk, with standard deviation ``parameter_std_rate`` times the
        square root of the elapsed time in seconds, the same model the
        estimator assumes through :meth:`parameter_process_psd`; then
        :meth:`update` draws the step's noise sample. The simulation calls
        this once per step before integrating the plant, so the integrator
        sees one frozen realisation per step, never a draw inside the solver.

        :param j2000: Current epoch in Julian centuries since J2000.
        """
        if self.estimated_vector_length > 0 and np.any(self.parameter_std_rate > 0.0):
            if not np.isfinite(self.last_update_time):
                self.last_update_time = j2000
            else:
                elapsed = (j2000 - self.last_update_time) * TimeConstants.cent2sec
                if elapsed > 0.0:
                    self.main_param = self.main_param + np.random.normal(
                        0.0, self.parameter_std_rate * np.sqrt(elapsed)
                    )
                    self.last_update_time = j2000
        self.update()

    def parameter_process_psd(self, *, form: str = "full") -> Covariance:
        r"""Return the continuous PSD of estimated disturbance parameters.

        ``parameter_std_rate`` is configuration, while ``std`` reports the
        current estimator uncertainty.  A disturbance with no estimated
        parameters naturally returns a zero-dimensional PSD.
        """
        return Covariance(
            np.diagflat(self.parameter_std_rate * self.parameter_std_rate),
            form=form,
            coordinates="disturbance_parameter_rate",
        )

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
