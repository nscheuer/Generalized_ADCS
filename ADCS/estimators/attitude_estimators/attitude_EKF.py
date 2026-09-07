r"""

.. container:: ekf-step ekf-step-input

   **1. Construct and validate the physical estimate**

   The constructor delegates to the shared estimator. It checks that the input
   is an :class:`~ADCS.state.EstimatorState`, that its covariance has full
   quaternion dimension, and that all augmented blocks are empty. The initial
   quaternion is normalized and the covariance is transformed with the
   corresponding normalization Jacobian.

   .. math::

      \hat{\mathbf{q}}_0 \leftarrow
      \frac{\mathbf{q}_0}{\lVert\mathbf{q}_0\rVert},
      \qquad
      \mathbf{P}_0 \leftarrow
      \mathbf{N}_0\mathbf{P}_{0,raw}\mathbf{N}_0^T,
      \qquad
      \mathbf{N}_0 =
      \frac{1}{\lVert\mathbf{q}_0\rVert}
      \left(\mathbf{I}_4-
      \frac{\mathbf{q}_0\mathbf{q}_0^T}{\lVert\mathbf{q}_0\rVert^2}\right).

   .. code-block:: python

      class EKF(AttitudeEstimator):
          def __init__(self, satellite, state, *, dt, ...):
              super().__init__(
                  satellite, state, dt=dt,
                  covariance_coordinates="full",
                  correction_mode="full_quaternion",
                  ...,
              )

.. container:: ekf-step ekf-step-predict

   **2. Propagate the nominal state**

   Given control :math:`\mathbf{u}_k`, orbital states at the beginning and end
   of the step, and :math:`\Delta t`, the deterministic model produces

   .. math::

      \hat{\mathbf{x}}_{k+1}^- =
      f(\hat{\mathbf{x}}_k^+,\mathbf{u}_k,
      \mathbf{o}_k,\mathbf{o}_{k+1},\Delta t).

   In code, :func:`~ADCS.estimators.process_model.propagate_state` delegates
   the physical integration to ``satellite.noiseless_rk4``. For an
   :class:`~ADCS.state.EstimatorState`, it propagates ``w``, ``q``, and ``h``
   while copying the empty parameter blocks and uncertainty containers.

   .. code-block:: python

      predicted = propagate_state(
          prior, self.satellite, control, step,
          orbital_state_start, orbital_state_end,
          midpoint_orbital_state=midpoint_orbital_state,
      )

.. container:: ekf-step ekf-step-linearize

   **3. Linearize the process model and discretize process noise**

   The continuous local error model is

   .. math::

      \delta\dot{\mathbf{x}} =
      \mathbf{F}_k\delta\mathbf{x} + \mathbf{w}_k,
      \qquad
      \mathbb{E}[\mathbf{w}_k\mathbf{w}_k^T] =
      \mathbf{Q}_{c,k}.

   The shared process-noise code obtains :math:`\mathbf{F}_k` from the
   spacecraft dynamics Jacobian and assembles :math:`\mathbf{Q}_{c,k}` from
   configured hardware noise plus ``unmodeled_dynamics_psd``. Van Loan
   discretization returns the transition matrix :math:`\mathbf{\Phi}_k` and
   discrete noise :math:`\mathbf{Q}_{d,k}`:

   .. math::

      \mathbf{\Phi}_k \approx e^{\mathbf{F}_k\Delta t},
      \qquad
      \mathbf{Q}_{d,k} =
      \int_0^{\Delta t}e^{\mathbf{F}_k\tau}
      \mathbf{Q}_{c,k}e^{\mathbf{F}_k^T\tau}\,d\tau.

   .. code-block:: python

      transition, process_noise = discretize_process_noise(
          prior, self.satellite, control, orbital_state_start, step,
          final_state=predicted,
          quaternion_mode="full_quaternion",
          quaternion_order="right",
          unmodeled_dynamics_psd=self.unmodeled_dynamics_psd,
      )

.. container:: ekf-step ekf-step-covariance

   **4. Predict the covariance**

   The :class:`~ADCS.covariance.Covariance` operation
   :meth:`~ADCS.covariance.Covariance.predicted_linear` applies

   .. math::

      \mathbf{P}_{k+1}^- =
      \mathbf{\Phi}_k\mathbf{P}_k^+\mathbf{\Phi}_k^T
      +\mathbf{Q}_{d,k}.

   The result is assigned to the predicted
   :class:`~ADCS.state.EstimatorState`, and the discrete process noise is kept
   alongside it for diagnostics and downstream consumers.

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

   **5. Select measurements and construct the innovation**

   :class:`~ADCS.estimators.measurement_stack.MeasurementStack` owns the
   canonical order, availability checks, sensor models, and wheel measurements.
   Raw measurements are filtered for finite values, enabled sources, and
   sampling schedules. The predicted measurement is

   .. math::

      \hat{\mathbf{z}}_k = h(\hat{\mathbf{x}}_k^-,\mathbf{o}_k).

   Additive measurements use :math:`\mathbf{r}=\mathbf{z}-\hat{\mathbf{z}}`.
   A quaternion measurement uses a right relative quaternion and a minimal
   three-coordinate map:

   .. math::

      \delta\mathbf{q}_{meas} =
      (\hat{\mathbf{q}}^-)^{-1}\otimes\mathbf{q}_{meas},
      \qquad
      \mathbf{r}_q = \phi(\delta\mathbf{q}_{meas})\in\mathbb{R}^3.

   Thus the residual can be three-dimensional even though the EKF covariance
   uses four stored quaternion coordinates.

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

   The active measurement Jacobian is expressed with respect to the full
   additive state coordinates:

   .. math::

      \mathbf{H}_k =
      \left.\frac{\partial h}{\partial\mathbf{x}}\right|_{\hat{\mathbf{x}}_k^-},
      \qquad
      \mathbf{R}_k =
      \mathbb{E}[\mathbf{v}_k\mathbf{v}_k^T].

   For quaternion measurements, the stack converts the four-coefficient
   measurement covariance into the same three-coordinate residual chart before
   assembling :math:`\mathbf{R}_k`. :class:`~ADCS.covariance.Covariance` keeps
   this measurement covariance in a validated block-diagonal object.

   .. code-block:: python

      measurement_jacobian = stack.jacobian(
          self._state, orbital_state, active,
          quaternion_mode=self.measurement_quaternion_mode,
          coordinates="full",
      )
      measurement_noise = stack.covariance(
          self._state, active,
          quaternion_mode=self.measurement_quaternion_mode,
      )

.. container:: ekf-step ekf-step-update

   **7. Compute the innovation covariance, gain, and correction**

   For :math:`m` active residual elements,

   .. math::

      \mathbf{S}_k =
      \mathbf{H}_k\mathbf{P}_k^-\mathbf{H}_k^T+\mathbf{R}_k,
      \qquad
      \mathbf{K}_k =
      \mathbf{P}_k^-\mathbf{H}_k^T\mathbf{S}_k^{-1},
      \qquad
      \delta\mathbf{x}_k = \mathbf{K}_k\mathbf{r}_k.

   The correction has full state dimension :math:`7+n_h`; its quaternion part
   has four additive components. The covariance uses the Joseph form:

   .. math::

      \mathbf{P}_{k,raw}^+ =
      (\mathbf{I}-\mathbf{K}\mathbf{H})\mathbf{P}^-
      (\mathbf{I}-\mathbf{K}\mathbf{H})^T
      +\mathbf{K}\mathbf{R}\mathbf{K}^T.

   .. code-block:: python

      gain, joseph_covariance = prior.covariance.updated_linear(
          measurement_jacobian, measurement_noise, joseph=True
      )
      correction = gain @ residual

.. container:: ekf-step ekf-step-reset

   **8. Retract the state, normalize the quaternion, and reset the covariance**

   The additive correction is applied by
   :meth:`~ADCS.state.State.plus` in the ``full_quaternion`` chart:

   .. math::

      \mathbf{x}_{k}^+ =
      \mathbf{x}_{k}^-\boxplus\delta\mathbf{x}_k,
      \qquad
      \mathbf{q}_{k}^+ =
      \frac{\hat{\mathbf{q}}_k^-+\delta\mathbf{q}_k}
      {\lVert\hat{\mathbf{q}}_k^-+\delta\mathbf{q}_k\rVert}.

   Because normalization changes the local linearization point, the covariance
   is transported with the reset Jacobian :math:`\mathbf{J}_{reset}`:

   .. math::

      \mathbf{P}_k^+ =
      \mathbf{J}_{reset}\mathbf{P}_{k,raw}^+
      \mathbf{J}_{reset}^T.

   The reset is identity for angular velocity and wheel momentum; only the
   quaternion block needs the chart-aware calculation owned by
   :class:`~ADCS.state.State`.

   .. code-block:: python

      reset_jacobian = prior.retraction_jacobian(
          correction, quaternion_mode="full_quaternion", quaternion_order="right"
      )
      corrected_covariance = prior.transport_covariance(
          joseph_covariance, correction,
          quaternion_mode="full_quaternion", quaternion_order="right"
      )
      corrected = prior.plus(
          correction, quaternion_mode="full_quaternion", quaternion_order="right"
      )
      corrected.covariance = corrected_covariance

"""

from __future__ import annotations

from typing import Any

from ADCS.state import EstimatorState, State

from .attitude_estimator import AttitudeEstimator


__all__ = ["EKF"]


class EKF(AttitudeEstimator):
    r"""Naive additive EKF with a normalized four-element quaternion block."""

    def __init__(
        self,
        satellite: Any,
        state: EstimatorState,
        *,
        dt: float,
        unmodeled_dynamics_psd: Any = 0.0,
        measurement_quaternion_mode: str = State.DEFAULT_QUATERNION_MODE,
    ) -> None:
        super().__init__(
            satellite,
            state,
            dt=dt,
            covariance_coordinates="full",
            correction_mode="full_quaternion",
            measurement_quaternion_mode=measurement_quaternion_mode,
            unmodeled_dynamics_psd=unmodeled_dynamics_psd,
        )
