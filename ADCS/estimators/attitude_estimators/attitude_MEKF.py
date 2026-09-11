r"""
.. container:: ekf-step ekf-step-input

   **1. Construct the tangent-space estimate**

   The MEKF estimates the physical state

   .. math::

      \hat{\mathbf{x}}_k =
      \begin{bmatrix}
      \hat{\boldsymbol{\omega}}_k\\
      \hat{\mathbf{q}}_k\\
      \hat{\mathbf{h}}_k
      \end{bmatrix},
      \qquad
      \delta\mathbf{x}_k =
      \begin{bmatrix}
      \delta\boldsymbol{\omega}_k\\
      \delta\boldsymbol{\theta}_k\\
      \delta\mathbf{h}_k
      \end{bmatrix}
      \in\mathbb{R}^{6+n_h}.

   :math:`\hat{\mathbf{q}}` remains a four-element unit quaternion in the
   nominal state, but its uncertainty is represented by the three-element
   right attitude error

   .. math::

      \delta\mathbf{q} =
      \hat{\mathbf{q}}^{-1}\otimes\mathbf{q},
      \qquad
      \delta\boldsymbol{\theta}=\phi(\delta\mathbf{q}).

   The :class:`~ADCS.state.EstimatorState` covariance is therefore stored by
   :class:`~ADCS.covariance.Covariance` in tangent coordinates. The selected
   ``quaternion_mode`` is ``quaternion_vector`` by default, or
   ``rotation_vector`` when requested; both are three-parameter charts.

   .. code-block:: python

      class MEKF(AttitudeEstimator):
          def __init__(self, satellite, state, *, dt, quaternion_mode, ...):
              super().__init__(
                  satellite, state, dt=dt,
                  covariance_coordinates="tangent",
                  correction_mode=quaternion_mode,
                  measurement_quaternion_mode=quaternion_mode,
                  ...,
              )

.. container:: ekf-step ekf-step-predict

   **2. Propagate the nominal state**

   The deterministic spacecraft model propagates the nominal physical state
   from :math:`t_k` to :math:`t_{k+1}`:

   .. math::

      \hat{\mathbf{x}}_{k+1}^- =
      f(\hat{\mathbf{x}}_k^+,\mathbf{u}_k,
      \mathbf{o}_k,\mathbf{o}_{k+1},\Delta t).

   The quaternion is integrated and normalized as part of the nominal state;
   the covariance remains in the three-dimensional local attitude chart.
   :func:`~ADCS.estimators.process_model.propagate_state` performs the
   physical propagation without mutating the prior estimate.

   .. code-block:: python

      predicted = propagate_state(
          prior, self.satellite, control, step,
          orbital_state_start, orbital_state_end,
          midpoint_orbital_state=midpoint_orbital_state,
      )

.. container:: ekf-step ekf-step-linearize

   **3. Linearize the tangent error model and discretize process noise**

   The local MEKF error evolves as

   .. math::

      \delta\dot{\mathbf{x}} =
      \mathbf{F}_k\delta\mathbf{x}+\mathbf{w}_k,
      \qquad
      \mathbb{E}[\mathbf{w}_k\mathbf{w}_k^T]=\mathbf{Q}_{c,k}.

   The process model maps the spacecraft dynamics into the selected tangent
   chart, including the motion of that chart as the nominal quaternion changes.
   Van Loan discretization produces

   .. math::

      \mathbf{\Phi}_k\approx e^{\mathbf{F}_k\Delta t},
      \qquad
      \mathbf{Q}_{d,k}=
      \int_0^{\Delta t}e^{\mathbf{F}_k\tau}
      \mathbf{Q}_{c,k}e^{\mathbf{F}_k^T\tau}\,d\tau.

   .. code-block:: python

      transition, process_noise = discretize_process_noise(
          prior, self.satellite, control, orbital_state_start, step,
          final_state=predicted,
          quaternion_mode=self.correction_mode,
          quaternion_order="right",
          unmodeled_dynamics_psd=self.unmodeled_dynamics_psd,
      )

.. container:: ekf-step ekf-step-covariance

   **4. Predict the tangent covariance**

   The :class:`~ADCS.covariance.Covariance` operation
   :meth:`~ADCS.covariance.Covariance.predicted_linear` applies the linearized
   covariance prediction directly in :math:`\mathbb{R}^{6+n_h}`:

   .. math::

      \mathbf{P}_{k+1}^- =
      \mathbf{\Phi}_k\mathbf{P}_k^+\mathbf{\Phi}_k^T
      +\mathbf{Q}_{d,k}.

   No four-element quaternion covariance is constructed during this step.

   .. code-block:: python

      predicted_covariance = prior.covariance.predicted_linear(
          transition, process_noise
      )
      predicted.covariance = predicted_covariance
      predicted.process_noise = Covariance(
          process_noise,
          form=prior.process_noise.form,
          coordinates=prior.process_noise.coordinates,
      )

.. container:: ekf-step ekf-step-measurement

   **5. Select measurements and construct the local innovation**

   :class:`~ADCS.estimators.measurement_stack.MeasurementStack` selects finite,
   enabled, and scheduled sources, then predicts the active measurements:

   .. math::

      \hat{\mathbf{z}}_k=h(\hat{\mathbf{x}}_k^-,\mathbf{o}_k).

   Additive measurements use :math:`\mathbf{r}=\mathbf{z}-\hat{\mathbf{z}}`.
   Quaternion measurements use a right relative quaternion and convert it into
   the selected three-coordinate chart:

   .. math::

      \delta\mathbf{q}_{meas}=
      (\hat{\mathbf{q}}^-)^{-1}\otimes\mathbf{q}_{meas},
      \qquad
      \mathbf{r}_q=\phi(\delta\mathbf{q}_{meas})\in\mathbb{R}^3.

   .. code-block:: python

      candidate = stack.active_mask(measurements, enabled=enabled, ...)
      predicted_measurements = stack.predict(
          self._state, orbital_state, active_mask=candidate
      )
      active = stack.active_mask(
          measurements, enabled=candidate,
          predicted=predicted_measurements, ...
      )
      residual = stack.residual(
          measurements, predicted_measurements, active,
          quaternion_mode=self.measurement_quaternion_mode,
      )

.. container:: ekf-step ekf-step-jacobian

   **6. Build** :math:`\mathbf{H}` **and** :math:`\mathbf{R}`

   The measurement Jacobian is expressed in the same tangent coordinates as
   the covariance:

   .. math::

      \mathbf{H}_k=
      \left.\frac{\partial h}{\partial\mathbf{x}}\right|_{\hat{\mathbf{x}}_k^-},
      \qquad
      \mathbf{R}_k=\mathbb{E}[\mathbf{v}_k\mathbf{v}_k^T].

   For a quaternion source, the measurement covariance is projected from the
   four stored coefficients into the same three-coordinate right-error chart.
   Thus :math:`\mathbf{H}` has tangent-state columns and quaternion residuals
   have three rows.

   .. code-block:: python

      measurement_jacobian = stack.jacobian(
          self._state, orbital_state, active,
          quaternion_mode=self.measurement_quaternion_mode,
          coordinates="tangent",
      )
      measurement_noise = stack.covariance(
          self._state, active,
          quaternion_mode=self.measurement_quaternion_mode,
      )

.. container:: ekf-step ekf-step-update

   **7. Compute the innovation covariance, gain, and tangent correction**

   For :math:`m` active residual elements,

   .. math::

      \mathbf{S}_k=\mathbf{H}_k\mathbf{P}_k^-\mathbf{H}_k^T+\mathbf{R}_k,
      \qquad
      \mathbf{K}_k=\mathbf{P}_k^-\mathbf{H}_k^T\mathbf{S}_k^{-1},
      \qquad
      \delta\mathbf{x}_k=\mathbf{K}_k\mathbf{r}_k.

   The correction has tangent-state dimension :math:`6+n_h`; its attitude
   block is a three-vector. The Joseph covariance update is

   .. math::

      \mathbf{P}_{k,raw}^+=
      (\mathbf{I}-\mathbf{K}\mathbf{H})\mathbf{P}^-
      (\mathbf{I}-\mathbf{K}\mathbf{H})^T
      +\mathbf{K}\mathbf{R}\mathbf{K}^T.

   .. code-block:: python

      gain, joseph_covariance = prior.covariance.updated_linear(
          measurement_jacobian, measurement_noise, joseph=True
      )
      correction = gain @ residual

.. container:: ekf-step ekf-step-reset

   **8. Apply the multiplicative correction and reset the covariance**

   The attitude part of :math:`\delta\mathbf{x}` is converted into a unit
   delta quaternion and composed on the right of the nominal quaternion:

   .. math::

      \mathbf{x}_k^+=\mathbf{x}_k^-\boxplus\delta\mathbf{x}_k,
      \qquad
      \mathbf{q}_k^+=\hat{\mathbf{q}}_k^-
      \otimes\phi^{-1}(\delta\boldsymbol{\theta}_k).

   Since the tangent origin has moved, the covariance is transported using the
   chart-specific reset Jacobian:

   .. math::

      \mathbf{P}_k^+=
      \mathbf{J}_{reset}\mathbf{P}_{k,raw}^+\mathbf{J}_{reset}^T.

   :meth:`~ADCS.state.State.plus` and
   :meth:`~ADCS.state.State.transport_covariance` own the quaternion chart
   operations, so the MEKF remains valid for either supported three-parameter
   ``quaternion_mode``.

   .. code-block:: python

      reset_jacobian = prior.retraction_jacobian(
          correction,
          quaternion_mode=self.correction_mode,
          quaternion_order="right",
      )
      corrected_covariance = prior.transport_covariance(
          joseph_covariance,
          correction,
          quaternion_mode=self.correction_mode,
          quaternion_order="right",
      )
      corrected = prior.plus(
          correction,
          quaternion_mode=self.correction_mode,
          quaternion_order="right",
      )
      corrected.covariance = corrected_covariance
"""

from __future__ import annotations

from typing import Any

from ADCS.state import EstimatorState, State

from .attitude_estimator import AttitudeEstimator


__all__ = ["MEKF"]


class MEKF(AttitudeEstimator):
    r"""Multiplicative EKF with a three-element right attitude error."""

    def __init__(
        self,
        satellite: Any,
        state: EstimatorState,
        *,
        dt: float,
        unmodeled_dynamics_psd: Any = 0.0,
        quaternion_mode: str = State.DEFAULT_QUATERNION_MODE,
    ) -> None:
        super().__init__(
            satellite,
            state,
            dt=dt,
            covariance_coordinates="tangent",
            correction_mode=quaternion_mode,
            measurement_quaternion_mode=quaternion_mode,
            unmodeled_dynamics_psd=unmodeled_dynamics_psd,
        )
