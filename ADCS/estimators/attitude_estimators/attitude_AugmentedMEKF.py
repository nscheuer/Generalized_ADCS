r"""Multiplicative EKF with joint bias and disturbance-parameter estimation.

.. container:: ekf-step ekf-step-input

   **1. Construct the augmented tangent-state estimate**

   Validate the physical state and append bias and disturbance-parameter
   blocks while retaining the three-coordinate right-attitude error.

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

   The nominal quaternion has four coefficients, while
   :math:`\delta\boldsymbol{\theta}` is its three-coordinate right error.

.. container:: ekf-step ekf-step-predict

   **2. Propagate the augmented nominal state**

   Synchronize the nominal augmented parameters into ``EstimatedSatellite``
   and propagate the nonlinear spacecraft model.

   .. math::

      \hat{\mathbf{x}}_{p,k+1}^{-} =
      f(\hat{\mathbf{x}}_{p,k}^{+},\hat{\mathbf{b}}_{a,k}^{+},
      \hat{\mathbf{d}}_k^{+},\mathbf{u}_k,\mathbf{o}_k,\mathbf{o}_{k+1},\Delta t),
      \qquad
      \begin{bmatrix}\hat{\mathbf{b}}_a\\\hat{\mathbf{b}}_s\\\hat{\mathbf{d}}\end{bmatrix}_{k+1}^{-}
      =
      \begin{bmatrix}\hat{\mathbf{b}}_a\\\hat{\mathbf{b}}_s\\\hat{\mathbf{d}}\end{bmatrix}_{k}^{+}.

.. container:: ekf-step ekf-step-linearize

   **3. Linearize the augmented process model**

   Construct the tangent error dynamics and discretize physical, bias, and
   disturbance-parameter process noise.

   .. math::

      \delta\dot{\mathbf{x}}_a=\mathbf{F}_{a,k}\delta\mathbf{x}_a+
      \mathbf{w}_{a,k},\qquad
      \mathbf{\Phi}_{a,k}\approx e^{\mathbf{F}_{a,k}\Delta t},
      \qquad
      \mathbf{Q}_{d,a,k}^{VL}=
      \int_0^{\Delta t}e^{\mathbf{F}_{a,k}\tau}\mathbf{Q}_{c,a,k}
      e^{\mathbf{F}_{a,k}^{T}\tau}\,d\tau.

   The parameter-error rows of :math:`\mathbf{F}_{a,k}` are zero for the
   deterministic random-walk model; their configured noise remains in
   :math:`\mathbf{Q}_{c,a,k}`.

.. container:: ekf-step ekf-step-covariance

   **4. Predict the tangent covariance**

   Apply the linear covariance recursion in the right-error coordinates.

   .. math::

      \mathbf{P}_{a,k+1}^{-}=
      \mathbf{\Phi}_{a,k}\mathbf{P}_{a,k}^{+}\mathbf{\Phi}_{a,k}^{T}
      +\mathbf{Q}_{d,a,k}^{model}+\mathbf{Q}_{d,a,k}^{state}.

.. container:: ekf-step ekf-step-measurement

   **5. Predict measurements and form the innovation**

   Use the shared measurement stack and its bias-aware residual definitions.

   .. math::

      \hat{\mathbf{z}}_k=h(\hat{\mathbf{x}}_{p,k}^{-},\mathbf{o}_k)
      +\mathbf{D}_{s,k}\hat{\mathbf{b}}_{s,k}^{-},
      \qquad
      \mathbf{r}_k=\mathbf{z}_k\boxminus\hat{\mathbf{z}}_k.

.. container:: ekf-step ekf-step-jacobian

   **6. Form the augmented measurement Jacobian**

   Include physical-state, sensor-bias, and disturbance-parameter sensitivities.

   .. math::

      \mathbf{H}_{a,k}=
      \begin{bmatrix}
      \dfrac{\partial h}{\partial\delta\mathbf{x}_p} &
      \dfrac{\partial h}{\partial\mathbf{b}_a} &
      \mathbf{D}_{s,k} &
      \dfrac{\partial h}{\partial\mathbf{d}}
      \end{bmatrix}_{\hat{\mathbf{x}}_{a,k}^{-}},
      \qquad
      \mathbf{R}_k=\mathbb{E}[\mathbf{v}_k\mathbf{v}_k^T].

.. container:: ekf-step ekf-step-update

   **7. Apply the Kalman correction**

   Correct the physical state and every enabled augmented parameter block.

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

   **8. Retract the attitude and reset the tangent covariance**

   Apply the multiplicative quaternion retraction and transport the covariance
   about the new tangent origin.

   .. math::

      \mathbf{x}_{a,k}^{+}=\mathbf{x}_{a,k}^{-}\boxplus\delta\mathbf{x}_{a,k},
      \qquad
      \mathbf{q}_k^{+}=\hat{\mathbf{q}}_k^{-}
      \otimes\phi^{-1}(\delta\boldsymbol{\theta}_k),
      \qquad
      \mathbf{P}_{a,k}^{+}=\mathbf{J}_{reset}\mathbf{P}_{a,k,raw}^{+}
      \mathbf{J}_{reset}^{T}.
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
