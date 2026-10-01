r"""Additive EKF with joint bias and disturbance-parameter estimation.

.. container:: ekf-step ekf-step-input

   **1. Construct the augmented full-state estimate**

   Validate the physical state and append the configured actuator-bias,
   sensor-bias, and disturbance-parameter blocks to the full-quaternion
   covariance layout.

   .. math::

      \hat{\mathbf{x}}_a =
      [\hat{\boldsymbol{\omega}},\hat{\mathbf{q}},\hat{\mathbf{h}},
      \hat{\mathbf{b}}_a,\hat{\mathbf{b}}_s,\hat{\mathbf{d}}]^T,
      \qquad
      \mathbf{P}_a\in\mathbb{R}^{n_a\times n_a},
      \qquad
      n_a=7+n_h+n_{b_a}+n_{b_s}+n_d.

   The additive covariance retains all four quaternion coefficients.

.. container:: ekf-step ekf-step-predict

   **2. Propagate the augmented nominal state**

   Synchronize the nominal augmented parameters into ``EstimatedSatellite``
   and propagate the spacecraft state through the nonlinear dynamics.

   .. math::

      \hat{\mathbf{x}}_{p,k+1}^{-} =
      f(\hat{\mathbf{x}}_{p,k}^{+},\hat{\mathbf{b}}_{a,k}^{+},
      \hat{\mathbf{d}}_k^{+},\mathbf{u}_k,\mathbf{o}_k,\mathbf{o}_{k+1},\Delta t),
      \qquad
      \begin{bmatrix}\hat{\mathbf{b}}_a\\\hat{\mathbf{b}}_s\\\hat{\mathbf{d}}\end{bmatrix}_{k+1}^{-}
      =
      \begin{bmatrix}\hat{\mathbf{b}}_a\\\hat{\mathbf{b}}_s\\\hat{\mathbf{d}}\end{bmatrix}_{k}^{+}.

   Here :math:`\mathbf{x}_p=[\boldsymbol{\omega},\mathbf{q},\mathbf{h}]^T`;
   the deterministic model holds parameter values constant.

.. container:: ekf-step ekf-step-linearize

   **3. Linearize the augmented process model**

   Build the physical, bias, and disturbance-parameter couplings and discretize
   the continuous process noise.

   .. math::

      \delta\dot{\mathbf{x}}_a=\mathbf{F}_{a,k}\delta\mathbf{x}_a+
      \mathbf{w}_{a,k},\qquad
      \mathbf{F}_{a,k}=
      \begin{bmatrix}
      \mathbf{F}_{pp} & \mathbf{F}_{p b_a} & \mathbf{0} & \mathbf{F}_{p d}\\
      \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{0}\\
      \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{0}\\
      \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{0}
      \end{bmatrix}.

   The zero lower rows implement random walks; their configured spectral
   densities are included in :math:`\mathbf{Q}_{c,a}` before discretization.

.. container:: ekf-step ekf-step-covariance

   **4. Predict the full covariance**

   Apply :math:`P^- = \\Phi P^+ \\Phi^T + Q_d` in full quaternion coordinates.

   .. math::

      \mathbf{P}_{a,k+1}^{-}=
      \mathbf{\Phi}_{a,k}\mathbf{P}_{a,k}^{+}\mathbf{\Phi}_{a,k}^{T}
      +\mathbf{Q}_{d,a,k}^{model}+\mathbf{Q}_{d,a,k}^{state}.

.. container:: ekf-step ekf-step-measurement

   **5. Predict measurements and form the innovation**

   Use the shared measurement stack, including bias terms in the augmented
   measurement model.

   .. math::

      \hat{\mathbf{z}}_k=h(\hat{\mathbf{x}}_{p,k}^{-},\mathbf{o}_k)
      +\mathbf{D}_{s,k}\hat{\mathbf{b}}_{s,k}^{-},
      \qquad
      \mathbf{r}_k=\mathbf{z}_k-\hat{\mathbf{z}}_k.

.. container:: ekf-step ekf-step-jacobian

   **6. Form the augmented measurement Jacobian**

   Include derivatives with respect to the physical state, sensor biases, and
   disturbance parameters.

   .. math::

      \mathbf{H}_{a,k}=
      \begin{bmatrix}
      \dfrac{\partial h}{\partial\mathbf{x}_p} &
      \dfrac{\partial h}{\partial\mathbf{b}_a} &
      \mathbf{D}_{s,k} &
      \dfrac{\partial h}{\partial\mathbf{d}}
      \end{bmatrix}_{\hat{\mathbf{x}}_{a,k}^{-}},
      \qquad
      \mathbf{R}_k=\mathbb{E}[\mathbf{v}_k\mathbf{v}_k^T].

.. container:: ekf-step ekf-step-update

   **7. Apply the Kalman correction**

   Update the complete augmented estimate and covariance.

   .. math::

      \mathbf{S}_k=\mathbf{H}_{a,k}\mathbf{P}_{a,k}^{-}\mathbf{H}_{a,k}^{T}+
      \mathbf{R}_k,\qquad
      \mathbf{K}_k=\mathbf{P}_{a,k}^{-}\mathbf{H}_{a,k}^{T}\mathbf{S}_k^{-1},
      \qquad
      \delta\mathbf{x}_{a,k}=\mathbf{K}_k\mathbf{r}_k.

      \mathbf{P}_{a,k,raw}^{+}=(\mathbf{I}-\mathbf{K}_k\mathbf{H}_{a,k})
      \mathbf{P}_{a,k}^{-}(\mathbf{I}-\mathbf{K}_k\mathbf{H}_{a,k})^T+
      \mathbf{K}_k\mathbf{R}_k\mathbf{K}_k^T.

.. container:: ekf-step ekf-step-reset

   **8. Normalize and retain the augmented state**

   Normalize the quaternion and retain the estimated parameter blocks for the
   next prediction.

   .. math::

      \mathbf{x}_{a,k}^{+}=\mathbf{x}_{a,k}^{-}\boxplus\delta\mathbf{x}_{a,k},
      \qquad
      \mathbf{q}_k^{+}=
      \frac{\hat{\mathbf{q}}_k^{-}+\delta\mathbf{q}_k}
      {\lVert\hat{\mathbf{q}}_k^{-}+\delta\mathbf{q}_k\rVert},
      \qquad
      \mathbf{P}_{a,k}^{+}=\mathbf{J}_{reset}\mathbf{P}_{a,k,raw}^{+}
      \mathbf{J}_{reset}^{T}.
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
