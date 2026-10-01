r"""Unscented attitude estimator with augmented parameter blocks.

.. container:: ekf-step ekf-step-input

   **1. Construct the augmented tangent-state estimate**

   Validate the physical, bias, and disturbance-parameter blocks and retain the
   right-error tangent covariance.

   .. math::

      \hat{\mathbf{x}}_a =
      [\hat{\boldsymbol{\omega}},\hat{\mathbf{q}},\hat{\mathbf{h}},
      \hat{\mathbf{b}}_a,\hat{\mathbf{b}}_s,\hat{\mathbf{d}}]^T,
      \qquad
      \delta\mathbf{x}_a =
      [\delta\boldsymbol{\omega},\delta\boldsymbol{\theta},\delta\mathbf{h},
      \delta\mathbf{b}_a,\delta\mathbf{b}_s,\delta\mathbf{d}]^T
      \in\mathbb{R}^{n_a},
      \qquad
      n_a=6+n_h+n_{b_a}+n_{b_s}+n_d.

   The covariance is in tangent coordinates; the nominal quaternion stays on
   :math:`\mathbb{S}^{3}`.

.. container:: ekf-step ekf-step-predict

   **2. Generate and propagate augmented sigma points**

   Each sigma point carries its own parameter realization; those values are
   synchronized into ``EstimatedSatellite`` before nonlinear propagation.

   .. math::

      \lambda=\alpha^2(n_a+\kappa)-n_a,\qquad
      \gamma=\sqrt{n_a+\lambda},\qquad
      \mathbf{P}_a=\mathbf{S}^{T}\mathbf{S},

      \boldsymbol{\xi}_0=\mathbf{0},\qquad
      \boldsymbol{\xi}_i=\pm\gamma\mathbf{S}_{i,:}^{T},\qquad
      \mathbf{X}_{i,k}^{+}=\hat{\mathbf{x}}_{a,k}^{+}
      \boxplus\boldsymbol{\xi}_i.

   The standard unscented mean and covariance weights
   :math:`W_i^{(m)}` and :math:`W_i^{(c)}` are computed with this augmented
   tangent dimension.

.. container:: ekf-step ekf-step-linearize

   **3. Form the unscented process statistics**

   Compute the weighted manifold mean and sigma deviations, then discretize and
   add the augmented process noise.

   .. math::

      \mathbf{X}_{i,k+1}^{-}=
      f_a(\mathbf{X}_{i,k}^{+},\mathbf{u}_k,\mathbf{o}_k,\mathbf{o}_{k+1},\Delta t),
      \qquad
      \sum_iW_i^{(m)}
      (\mathbf{X}_{i,k+1}^{-}\boxminus\hat{\mathbf{x}}_{a,k+1}^{-})=\mathbf{0}.

      \mathbf{d}_{i,k}^{x}=\mathbf{X}_{i,k+1}^{-}
      \boxminus\hat{\mathbf{x}}_{a,k+1}^{-},\qquad
      \mathbf{P}_{a,k+1}^{-}=
      \sum_iW_i^{(c)}\mathbf{d}_{i,k}^{x}(\mathbf{d}_{i,k}^{x})^{T}
      +\mathbf{Q}_{d,a,k}^{model}+\mathbf{Q}_{d,a,k}^{state}.

   The augmented parameter values are held by each deterministic propagation;
   their uncertainty enters through the state and model process covariances.

.. container:: ekf-step ekf-step-measurement

   **4. Predict measurements for every sigma point**

   Transform sigma points through the shared measurement stack and form
   measurement and state cross-covariances.

   .. math::

      \mathbf{Z}_{i,k}=h(\mathbf{X}_{i,k}^{-},\mathbf{o}_k),\qquad
      \mathbf{d}_{i,k}^{z}=\mathbf{Z}_{i,k}\boxminus\hat{\mathbf{z}}_k,

      \mathbf{P}_{zz}=\sum_iW_i^{(c)}\mathbf{d}_{i,k}^{z}
      (\mathbf{d}_{i,k}^{z})^{T}+\mathbf{R}_k,\qquad
      \mathbf{P}_{xz}=\sum_iW_i^{(c)}\mathbf{d}_{i,k}^{x}
      (\mathbf{d}_{i,k}^{z})^{T}.

.. container:: ekf-step ekf-step-update

   **5. Apply the unscented correction**

   Compute the gain, update the physical state and augmented parameters, and
   transport the posterior covariance.

   .. math::

      \mathbf{K}_k=\mathbf{P}_{xz}\mathbf{P}_{zz}^{-1},\qquad
      \mathbf{r}_k=\mathbf{z}_k\boxminus\hat{\mathbf{z}}_k,\qquad
      \delta\mathbf{x}_{a,k}=\mathbf{K}_k\mathbf{r}_k,

      \mathbf{P}_{a,k,raw}^{+}=\mathbf{P}_{a,k}^{-}-
      \mathbf{K}_k\mathbf{P}_{zz}\mathbf{K}_k^{T}.

.. container:: ekf-step ekf-step-reset

   **6. Retract the attitude and retain the augmented estimate**

   Keep the right-error tangent chart and preserve all bias and disturbance
   parameter blocks for the next sigma-point generation.

   .. math::

      \mathbf{x}_{a,k}^{+}=\mathbf{x}_{a,k}^{-}\boxplus\delta\mathbf{x}_{a,k},
      \qquad
      \mathbf{q}_k^{+}=\hat{\mathbf{q}}_k^{-}
      \otimes\phi^{-1}(\delta\boldsymbol{\theta}_k),
      \qquad
      \mathbf{P}_{a,k}^{+}=\mathbf{J}_{reset}\mathbf{P}_{a,k,raw}^{+}
      \mathbf{J}_{reset}^{T}.
"""

from .attitude_UKF import UKF

__all__ = ["AugmentedUKF"]


class AugmentedUKF(UKF):
    """UKF supporting sensor-bias and disturbance-parameter state blocks."""

    supports_augmented_parameters = True
