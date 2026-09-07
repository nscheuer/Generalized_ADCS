r"""
.. container:: ekf-step ekf-step-input

   **1. Construct the square-root tangent-state filter**

   SRUKF uses the same non-augmented physical state and right tangent attitude
   error as :class:`~ADCS.estimators.attitude_estimators.attitude_UKF.UKF`:

   .. math::

      \hat{\mathbf{x}}_k =
      [\hat{\boldsymbol{\omega}}_k,\hat{\mathbf{q}}_k,\hat{\mathbf{h}}_k]^T,
      \qquad
      \delta\mathbf{x}_k =
      [\delta\boldsymbol{\omega}_k,
      \delta\boldsymbol{\theta}_k,\delta\mathbf{h}_k]^T
      \in\mathbb{R}^{6+n_h}.

   The defining difference is the covariance representation. Instead of
   storing :math:`\mathbf{P}` directly, :class:`~ADCS.covariance.Covariance`
   stores an upper factor :math:`\mathbf{S}` such that

   .. math::

      \mathbf{P}=\mathbf{S}^T\mathbf{S}.

   The constructor converts both the state covariance and process-noise
   covariance to ``form="sqrt"`` before entering the shared UKF lifecycle.

   .. code-block:: python

      result = state.copy()
      result.covariance = state.covariance.copy(form="sqrt")
      result.process_noise = state.process_noise.copy(form="sqrt")
      super().__init__(
          satellite, result, dt=dt,
          quaternion_mode=quaternion_mode,
          alpha=alpha, beta=beta, kappa=kappa,
          ...,
      )

.. container:: ekf-step ekf-step-predict

   **2. Generate tangent sigma points from the square-root factor**

   For tangent dimension :math:`n=6+n_h`,

   .. math::

      \lambda=\alpha^2(n+\kappa)-n,
      \qquad
      \gamma=\sqrt{n+\lambda},
      \qquad
      \mathbf{P}=\mathbf{S}^T\mathbf{S}.

   The sigma offsets are rows of the upper factor:

   .. math::

      \boldsymbol{\xi}_0=\mathbf{0},
      \qquad
      \boldsymbol{\xi}_i=\pm\gamma\,\mathbf{S}_{i,:}^T,
      \qquad
      \mathbf{X}_i=\hat{\mathbf{x}}\boxplus\boldsymbol{\xi}_i.

   The mean and covariance weights are the standard unscented weights. No
   square-root factor is expanded into a full covariance to generate points.

   .. code-block:: python

      points, offsets, mean_weights, covariance_weights = self._sigma_states(
          prior
      )

.. container:: ekf-step ekf-step-linearize

   **3. Propagate sigma points and form the prediction**

   Each sigma point is independently propagated:

   .. math::

      \mathbf{X}_{i,k+1}^- =
      f(\mathbf{X}_{i,k}^+,\mathbf{u}_k,
      \mathbf{o}_k,\mathbf{o}_{k+1},\Delta t).

   As in the UKF, process noise is constructed separately and added after the
   propagated sigma-point deviations have been computed.

   .. code-block:: python

      propagated_points = [
          propagate_state(
              point, self.satellite, control, step,
              orbital_state_start, orbital_state_end,
              midpoint_orbital_state=midpoint_orbital_state,
          )
          for point in points
      ]

   After propagation, compute the manifold mean and square-root predicted covariance.

   The predicted state is the weighted manifold mean:

   .. math::

      \sum_i W_i^{(m)}
      (\mathbf{X}_{i,k+1}^-\boxminus\hat{\mathbf{x}}_{k+1}^-)=\mathbf{0}.

   With :math:`\mathbf{d}_{i,k}^x` denoting the local sigma deviations, the
   covariance is mathematically

   .. math::

      \mathbf{P}_{k+1}^- =
      \sum_i W_i^{(c)}\mathbf{d}_{i,k}^x(\mathbf{d}_{i,k}^x)^T
      +\mathbf{Q}_{d,k}.

   In SRUKF, :class:`~ADCS.covariance.Covariance` forms its upper factor using
   QR factorization and rank updates/downdates, preserving
   :math:`\mathbf{P}=\mathbf{S}^T\mathbf{S}` as the public representation.

   .. code-block:: python

      predicted = self._state_mean(propagated_points, mean_weights)
      transition, process_noise = discretize_process_noise(
          prior, self.satellite, control, orbital_state_start, step,
          final_state=predicted,
          quaternion_mode=self.correction_mode,
          quaternion_order="right",
          unmodeled_dynamics_psd=self.unmodeled_dynamics_psd,
      )
      deviations = np.vstack([
          point.minus(predicted,
              quaternion_mode=self.correction_mode,
              quaternion_order="right")
          for point in propagated_points
      ])
      predicted.covariance = prior.covariance.predicted_unscented(
          deviations, covariance_weights, process_noise
      )

.. container:: ekf-step ekf-step-measurement

   **4. Predict measurements and form their statistics**

   Active sigma points are passed through the measurement stack:

   .. math::

      \mathbf{Z}_{i,k}=h(\mathbf{X}_{i,k}^-,\mathbf{o}_k).

   Additive measurements use a weighted Euclidean mean. Quaternion
   measurements use a weighted right-error manifold mean in the selected
   three-parameter chart:

   .. math::

      \sum_i W_i^{(m)}
      (\mathbf{Z}_{i,k}\boxminus\hat{\mathbf{z}}_k)=\mathbf{0}.

   .. code-block:: python

      sigma_measurements = [
          stack.predict(point, orbital_state, active_mask=active)
          for point in points
      ]
      predicted_measurement = self._measurement_mean(
          stack, sigma_measurements, active, mean_weights
      )

   Then form :math:`\mathbf{P}_{zz}` and :math:`\mathbf{P}_{xz}` from the
   measurement and state deviations.

   Measurement deviations are expressed in residual coordinates and state
   deviations in tangent coordinates:

   .. math::

      \mathbf{d}_{i,k}^z=\mathbf{Z}_{i,k}\boxminus\hat{\mathbf{z}}_k,
      \qquad
      \mathbf{P}_{zz}=\sum_i W_i^{(c)}\mathbf{d}_{i,k}^z
      (\mathbf{d}_{i,k}^z)^T+\mathbf{R}_k,

   .. math::

      \mathbf{P}_{xz}=\sum_i W_i^{(c)}\mathbf{d}_{i,k}^x
      (\mathbf{d}_{i,k}^z)^T.

   The square-root form computes the measurement covariance using QR and
   weighted Cholesky updates where needed; no measurement Jacobian is formed.

   .. code-block:: python

      measurement_deviations = np.vstack([
          stack.residual(
              measurement, predicted_measurement, active,
              quaternion_mode=self.measurement_quaternion_mode,
          )
          for measurement in sigma_measurements
      ])
      measurement_noise = stack.covariance(
          prior, active,
          quaternion_mode=self.measurement_quaternion_mode,
      )

.. container:: ekf-step ekf-step-update

   **5. Compute the square-root unscented gain and correction**

   The gain is obtained from the weighted cross-covariance and measurement
   covariance:

   .. math::

      \mathbf{K}_k=\mathbf{P}_{xz}\mathbf{P}_{zz}^{-1},
      \qquad
      \mathbf{r}_k=\mathbf{z}_k\boxminus\hat{\mathbf{z}}_k,
      \qquad
      \delta\mathbf{x}_k=\mathbf{K}_k\mathbf{r}_k.

   ``updated_unscented`` solves with the stored square-root representation and
   returns the posterior covariance in square-root form.

   .. code-block:: python

      gain, posterior_covariance = prior.covariance.updated_unscented(
          state_deviations,
          measurement_deviations,
          covariance_weights,
          measurement_noise,
      )
      innovation = stack.residual(
          measurements, predicted_measurement, active,
          quaternion_mode=self.measurement_quaternion_mode,
      )
      correction = gain @ innovation

.. container:: ekf-step ekf-step-reset

   **6. Retract the state and preserve square-root covariance storage**

   The tangent correction is applied as a right multiplicative quaternion
   update:

   .. math::

      \mathbf{x}_k^+=\mathbf{x}_k^-\boxplus\delta\mathbf{x}_k,
      \qquad
      \mathbf{q}_k^+=\hat{\mathbf{q}}_k^-
      \otimes\phi^{-1}(\delta\boldsymbol{\theta}_k).

   The posterior covariance is transported to the new tangent origin. Because
   the input covariance is square-root form, the transformed result preserves
   the upper-factor representation:

   .. math::

      \mathbf{P}_k^+=
      \mathbf{J}_{reset}\mathbf{P}_{k,raw}^+\mathbf{J}_{reset}^T,
      \qquad
      \mathbf{P}_k^+=\mathbf{S}_k^{+T}\mathbf{S}_k^+.

   .. code-block:: python

      corrected = prior.plus(
          correction,
          quaternion_mode=self.correction_mode,
          quaternion_order="right",
      )
      corrected.covariance = prior.transport_covariance(
          posterior_covariance,
          correction,
          quaternion_mode=self.correction_mode,
          quaternion_order="right",
      )
"""

from __future__ import annotations

from typing import Any

from ADCS.state import EstimatorState, State

from .attitude_UKF import UKF


__all__ = ["SRUKF"]


class SRUKF(UKF):
    r"""Non-augmented UKF that stores its state covariance as an upper factor.

    This filter uses the same tangent-state sigma points, manifold means, and
    additive process and measurement noise as :class:`UKF`. Its covariance is
    always retained in ``Covariance(form="sqrt")`` form, so the shared
    unscented covariance operations use square-root QR updates.
    """

    def __init__(
        self,
        satellite: Any,
        state: EstimatorState,
        *,
        dt: float,
        unmodeled_dynamics_psd: Any = 0.0,
        quaternion_mode: str = State.DEFAULT_QUATERNION_MODE,
        alpha: float = 1.0,
        beta: float = 2.0,
        kappa: float = 0.0,
    ) -> None:
        super().__init__(
            satellite,
            self._square_root_state(state),
            dt=dt,
            unmodeled_dynamics_psd=unmodeled_dynamics_psd,
            quaternion_mode=quaternion_mode,
            alpha=alpha,
            beta=beta,
            kappa=kappa,
        )

    @staticmethod
    def _square_root_state(state: EstimatorState) -> EstimatorState:
        if not isinstance(state, EstimatorState):
            raise TypeError(f"state must be an EstimatorState, got {type(state).__name__}")
        result = state.copy()
        result.covariance = state.covariance.copy(form="sqrt")
        result.process_noise = state.process_noise.copy(form="sqrt")
        return result

    def reset(self, state: EstimatorState) -> EstimatorState:
        """Reset while preserving square-root covariance storage."""
        return super().reset(self._square_root_state(state))
