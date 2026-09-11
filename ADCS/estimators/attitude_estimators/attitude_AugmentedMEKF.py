r"""Multiplicative EKF with joint bias and disturbance-parameter estimation.

.. container:: ekf-step ekf-step-input

   **1. Construct the augmented tangent-state estimate**

   Validate the physical state and append bias and disturbance-parameter
   blocks while retaining the three-coordinate right-attitude error.

.. container:: ekf-step ekf-step-predict

   **2. Propagate the augmented nominal state**

   Synchronize the nominal augmented parameters into ``EstimatedSatellite``
   and propagate the nonlinear spacecraft model.

.. container:: ekf-step ekf-step-linearize

   **3. Linearize the augmented process model**

   Construct the tangent error dynamics and discretize physical, bias, and
   disturbance-parameter process noise.

.. container:: ekf-step ekf-step-covariance

   **4. Predict the tangent covariance**

   Apply the linear covariance recursion in the right-error coordinates.

.. container:: ekf-step ekf-step-measurement

   **5. Predict measurements and form the innovation**

   Use the shared measurement stack and its bias-aware residual definitions.

.. container:: ekf-step ekf-step-jacobian

   **6. Form the augmented measurement Jacobian**

   Include physical-state, sensor-bias, and disturbance-parameter sensitivities.

.. container:: ekf-step ekf-step-update

   **7. Apply the Kalman correction**

   Correct the physical state and every enabled augmented parameter block.

.. container:: ekf-step ekf-step-reset

   **8. Retract the attitude and reset the tangent covariance**

   Apply the multiplicative quaternion retraction and transport the covariance
   about the new tangent origin.
"""

from __future__ import annotations

from .attitude_MEKF import MEKF


__all__ = ["AugmentedMEKF"]


class AugmentedMEKF(MEKF):
    """MEKF for the full :class:`~ADCS.state.EstimatorState` layout.

    The state may contain actuator biases, attitude-sensor biases, and
    estimated disturbance parameters in addition to angular velocity,
    attitude, and reaction-wheel momentum. Their dynamics and measurement
    couplings are supplied by ``EstimatedSatellite`` and the shared process
    model/measurement stack; the covariance remains in right tangent
    coordinates, just as for :class:`MEKF`.
    """

    supports_augmented_parameters = True
