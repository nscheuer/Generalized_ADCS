r"""Square-root unscented attitude estimator with augmented parameter blocks.

.. container:: ekf-step ekf-step-input

   **1. Construct the augmented square-root estimate**

   Validate the augmented tangent-state layout and store its covariance as an
   upper square-root factor.

.. container:: ekf-step ekf-step-predict

   **2. Generate and propagate augmented sigma points**

   Synchronize each sigma point's bias and disturbance parameters into
   ``EstimatedSatellite`` before nonlinear propagation.

.. container:: ekf-step ekf-step-linearize

   **3. Form square-root process statistics**

   Compute the weighted manifold mean and use QR/rank updates to incorporate
   augmented process noise into the predicted square-root covariance.

.. container:: ekf-step ekf-step-measurement

   **4. Predict sigma-point measurements**

   Form measurement deviations and state cross-covariances through the shared
   measurement stack.

.. container:: ekf-step ekf-step-update

   **5. Apply the square-root unscented correction**

   Compute the gain and update the physical and augmented state blocks while
   preserving the square-root covariance representation.

.. container:: ekf-step ekf-step-reset

   **6. Retract the attitude and retain the augmented estimate**

   Keep the tangent chart and the square-root covariance for the next cycle.
"""

from .attitude_SRUKF import SRUKF

__all__ = ["AugmentedSRUKF"]


class AugmentedSRUKF(SRUKF):
    """SRUKF supporting sensor-bias and disturbance-parameter state blocks."""

    supports_augmented_parameters = True
