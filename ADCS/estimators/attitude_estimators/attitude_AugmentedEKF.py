r"""Additive EKF with joint bias and disturbance-parameter estimation.

.. container:: ekf-step ekf-step-input

   **1. Construct the augmented full-state estimate**

   Validate the physical state and append the configured actuator-bias,
   sensor-bias, and disturbance-parameter blocks to the full-quaternion
   covariance layout.

.. container:: ekf-step ekf-step-predict

   **2. Propagate the augmented nominal state**

   Synchronize the nominal augmented parameters into ``EstimatedSatellite``
   and propagate the spacecraft state through the nonlinear dynamics.

.. container:: ekf-step ekf-step-linearize

   **3. Linearize the augmented process model**

   Build the physical, bias, and disturbance-parameter couplings and discretize
   the continuous process noise.

.. container:: ekf-step ekf-step-covariance

   **4. Predict the full covariance**

   Apply :math:`P^- = \\Phi P^+ \\Phi^T + Q_d` in full quaternion coordinates.

.. container:: ekf-step ekf-step-measurement

   **5. Predict measurements and form the innovation**

   Use the shared measurement stack, including bias terms in the augmented
   measurement model.

.. container:: ekf-step ekf-step-jacobian

   **6. Form the augmented measurement Jacobian**

   Include derivatives with respect to the physical state, sensor biases, and
   disturbance parameters.

.. container:: ekf-step ekf-step-update

   **7. Apply the Kalman correction**

   Update the complete augmented estimate and covariance.

.. container:: ekf-step ekf-step-reset

   **8. Normalize and retain the augmented state**

   Normalize the quaternion and retain the estimated parameter blocks for the
   next prediction.
"""

from __future__ import annotations

from .attitude_EKF import EKF


__all__ = ["AugmentedEKF"]


class AugmentedEKF(EKF):
    """Additive EKF for the full :class:`~ADCS.state.EstimatorState` layout.

    With empty augmented blocks this is equivalent to the existing ``EKF``.
    When the estimated satellite and state provide bias or disturbance
    parameter blocks, the shared full-quaternion covariance includes them in
    the prediction and measurement update.
    """

    supports_augmented_parameters = True
