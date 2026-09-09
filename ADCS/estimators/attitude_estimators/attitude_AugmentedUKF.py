r"""Unscented attitude estimator with augmented parameter blocks.

.. container:: ekf-step ekf-step-input

   **1. Construct the augmented tangent-state estimate**

   Validate the physical, bias, and disturbance-parameter blocks and retain the
   right-error tangent covariance.

.. container:: ekf-step ekf-step-predict

   **2. Generate and propagate augmented sigma points**

   Each sigma point carries its own parameter realization; those values are
   synchronized into ``EstimatedSatellite`` before nonlinear propagation.

.. container:: ekf-step ekf-step-linearize

   **3. Form the unscented process statistics**

   Compute the weighted manifold mean and sigma deviations, then discretize and
   add the augmented process noise.

.. container:: ekf-step ekf-step-measurement

   **4. Predict measurements for every sigma point**

   Transform sigma points through the shared measurement stack and form
   measurement and state cross-covariances.

.. container:: ekf-step ekf-step-update

   **5. Apply the unscented correction**

   Compute the gain, update the physical state and augmented parameters, and
   transport the posterior covariance.

.. container:: ekf-step ekf-step-reset

   **6. Retract the attitude and retain the augmented estimate**

   Keep the right-error tangent chart and preserve all bias and disturbance
   parameter blocks for the next sigma-point generation.
"""

from .attitude_UKF import UKF

__all__ = ["AugmentedUKF"]


class AugmentedUKF(UKF):
    """UKF supporting sensor-bias and disturbance-parameter state blocks."""

    supports_augmented_parameters = True
