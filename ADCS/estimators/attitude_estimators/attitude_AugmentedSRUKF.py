r"""Square-root unscented attitude estimator with augmented parameter blocks.

.. container:: ekf-step ekf-step-input

   **1. Construct the augmented square-root estimate**

   Validate the augmented tangent-state layout and store its covariance as an
   upper square-root factor.

   .. math::

      \hat{\mathbf{x}}_a =
      [\hat{\boldsymbol{\omega}},\hat{\mathbf{q}},\hat{\mathbf{h}},
      \hat{\mathbf{b}}_a,\hat{\mathbf{b}}_s,\hat{\mathbf{d}}]^T,
      \qquad
      \delta\mathbf{x}_a\in\mathbb{R}^{n_a},
      \qquad
      n_a=6+n_h+n_{b_a}+n_{b_s}+n_d,

      \mathbf{P}_a=\mathbf{S}_a^{T}\mathbf{S}_a.

   The nominal quaternion is retained separately from its three-coordinate
   right-attitude error in :math:`\delta\mathbf{x}_a`.

.. container:: ekf-step ekf-step-predict

   **2. Generate and propagate augmented sigma points**

   Synchronize each sigma point's bias and disturbance parameters into
   ``EstimatedSatellite`` before nonlinear propagation.

   .. math::

      \lambda=\alpha^2(n_a+\kappa)-n_a,\qquad
      \gamma=\sqrt{n_a+\lambda},\qquad
      \boldsymbol{\xi}_i=\pm\gamma\mathbf{S}_{a,i,:}^{T},
      \qquad
      \mathbf{X}_{i,k}^{+}=\hat{\mathbf{x}}_{a,k}^{+}
      \boxplus\boldsymbol{\xi}_i.

   The central point has :math:`\boldsymbol{\xi}_0=\mathbf{0}` and all weights
   use the augmented dimension :math:`n_a`.

.. container:: ekf-step ekf-step-linearize

   **3. Form square-root process statistics**

   Compute the weighted manifold mean and use QR/rank updates to incorporate
   augmented process noise into the predicted square-root covariance.

   .. math::

      \mathbf{X}_{i,k+1}^{-}=
      f_a(\mathbf{X}_{i,k}^{+},\mathbf{u}_k,\mathbf{o}_k,\mathbf{o}_{k+1},\Delta t),
      \qquad
      \sum_iW_i^{(m)}
      (\mathbf{X}_{i,k+1}^{-}\boxminus\hat{\mathbf{x}}_{a,k+1}^{-})=\mathbf{0}.

      \mathbf{P}_{a,k+1}^{-}=
      \sum_iW_i^{(c)}\mathbf{d}_{i,k}^{x}(\mathbf{d}_{i,k}^{x})^{T}
      +\mathbf{Q}_{d,a,k}^{model}+\mathbf{Q}_{d,a,k}^{state},
      \qquad
      \mathbf{P}_{a,k+1}^{-}=\mathbf{S}_{a,k+1}^{-T}\mathbf{S}_{a,k+1}^{-}.

   QR factorization and Cholesky rank updates/downdates construct
   :math:`\mathbf{S}_{a,k+1}^{-}` without storing a full posterior covariance.

.. container:: ekf-step ekf-step-measurement

   **4. Predict sigma-point measurements**

   Form measurement deviations and state cross-covariances through the shared
   measurement stack.

   .. math::

      \mathbf{Z}_{i,k}=h(\mathbf{X}_{i,k}^{-},\mathbf{o}_k),\qquad
      \mathbf{d}_{i,k}^{z}=\mathbf{Z}_{i,k}\boxminus\hat{\mathbf{z}}_k,

      \mathbf{P}_{zz}=\sum_iW_i^{(c)}\mathbf{d}_{i,k}^{z}
      (\mathbf{d}_{i,k}^{z})^{T}+\mathbf{R}_k,\qquad
      \mathbf{P}_{xz}=\sum_iW_i^{(c)}\mathbf{d}_{i,k}^{x}
      (\mathbf{d}_{i,k}^{z})^{T}.

.. container:: ekf-step ekf-step-update

   **5. Apply the square-root unscented correction**

   Compute the gain and update the physical and augmented state blocks while
   preserving the square-root covariance representation.

   .. math::

      \mathbf{K}_k=\mathbf{P}_{xz}\mathbf{P}_{zz}^{-1},\qquad
      \delta\mathbf{x}_{a,k}=\mathbf{K}_k
      (\mathbf{z}_k\boxminus\hat{\mathbf{z}}_k),

      \mathbf{P}_{a,k,raw}^{+}=\mathbf{P}_{a,k}^{-}-
      \mathbf{K}_k\mathbf{P}_{zz}\mathbf{K}_k^{T}
      =\mathbf{S}_{a,k,raw}^{+T}\mathbf{S}_{a,k,raw}^{+}.

.. container:: ekf-step ekf-step-reset

   **6. Retract the attitude and retain the augmented estimate**

   Keep the tangent chart and the square-root covariance for the next cycle.

   .. math::

      \mathbf{x}_{a,k}^{+}=\mathbf{x}_{a,k}^{-}\boxplus\delta\mathbf{x}_{a,k},
      \qquad
      \mathbf{q}_k^{+}=\hat{\mathbf{q}}_k^{-}
      \otimes\phi^{-1}(\delta\boldsymbol{\theta}_k),

      \mathbf{P}_{a,k}^{+}=\mathbf{J}_{reset}\mathbf{P}_{a,k,raw}^{+}
      \mathbf{J}_{reset}^{T}
      =\mathbf{S}_{a,k}^{+T}\mathbf{S}_{a,k}^{+}.
"""

from .attitude_SRUKF import SRUKF

__all__ = ["AugmentedSRUKF"]


class AugmentedSRUKF(SRUKF):
    """SRUKF supporting sensor-bias and disturbance-parameter state blocks."""

    supports_augmented_parameters = True
