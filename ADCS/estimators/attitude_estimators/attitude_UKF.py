r"""
.. container:: ekf-step ekf-step-input

   **1. Construct the non-augmented tangent-state filter**

   The UKF estimates only the physical state. Its nominal quaternion remains a
   four-element unit quaternion, while the uncertainty is represented in the
   right tangent coordinates

   .. math::

      \hat{\mathbf{x}}_k =
      [\hat{\boldsymbol{\omega}}_k,\hat{\mathbf{q}}_k,\hat{\mathbf{h}}_k]^T,
      \qquad
      \delta\mathbf{x}_k =
      [\delta\boldsymbol{\omega}_k,
      \delta\boldsymbol{\theta}_k,\delta\mathbf{h}_k]^T
      \in\mathbb{R}^{6+n_h}.

   Process, sensor-bias, actuator-bias, and disturbance variables are not
   appended to the sigma-point state.  When actuator input noise is nonzero,
   its non-perfect command channels are appended only for prediction. The
   covariance is owned by
   :class:`~ADCS.covariance.Covariance` in tangent coordinates. The chart is
   selected by ``quaternion_mode`` and is ``quaternion_vector`` by default.

   .. code-block:: python

      super().__init__(
          satellite, state, dt=dt,
          covariance_coordinates="tangent",
          correction_mode=quaternion_mode,
          measurement_quaternion_mode=quaternion_mode,
          ...,
      )

.. container:: ekf-step ekf-step-predict

   **2. Generate tangent sigma points and weights**

   For covariance dimension :math:`n=6+n_h`, define

   .. math::

      \lambda=\alpha^2(n+\kappa)-n,
      \qquad
      \gamma=\sqrt{n+\lambda}.

   With :math:`\mathbf{S}` such that
   :math:`\mathbf{P}=\mathbf{S}^T\mathbf{S}`, the sigma offsets are

   .. math::

      \boldsymbol{\xi}_0=\mathbf{0},
      \qquad
      \boldsymbol{\xi}_i=\pm\gamma\,\mathbf{S}_{i,:}^T,
      \qquad
      \mathbf{X}_i=\hat{\mathbf{x}}\boxplus\boldsymbol{\xi}_i.

   The mean and covariance weights are

   .. math::

      W_0^{(m)}=\frac{\lambda}{n+\lambda},
      \quad
      W_0^{(c)}=W_0^{(m)}+1-\alpha^2+\beta,
      \quad
      W_i^{(m)}=W_i^{(c)}=\frac{1}{2(n+\lambda)}.

   .. code-block:: python

      points, offsets, mean_weights, covariance_weights = self._sigma_states(
          prior
      )

.. container:: ekf-step ekf-step-linearize

   **3. Propagate sigma points and form the prediction**

   Each sigma state is propagated through the nonlinear spacecraft model.  If
   actuator input noise is configured, control-noise sigma points use a
   perturbed command; otherwise every point uses the nominal control:

   .. math::

      \mathbf{X}_{i,k+1}^- =
      f(\mathbf{X}_{i,k}^+,\mathbf{u}_k+\delta\mathbf{u}_i,
      \mathbf{o}_k,\mathbf{o}_{k+1},\Delta t).

   There is no state Jacobian in this step. The nonlinear model is evaluated at
   every sigma point, and remaining additive process noise is added after the
   propagated-point statistics have been formed.

   .. code-block:: python

      propagated_points = [
          propagate_state(
              point, self.satellite, control, step,
              orbital_state_start, orbital_state_end,
              midpoint_orbital_state=midpoint_orbital_state,
          )
          for point in points
      ]

   After propagation, compute the manifold state mean and predicted covariance.

   The predicted mean is the weighted manifold mean, found by iterating until
   the weighted local correction vanishes:

   .. math::

      \hat{\mathbf{x}}_{k+1}^- \text{ satisfies }
      \sum_i W_i^{(m)}
      (\mathbf{X}_{i,k+1}^-\boxminus\hat{\mathbf{x}}_{k+1}^-)=\mathbf{0}.

   The sigma deviations and discretized process noise then give

   .. math::

      \mathbf{d}_{i,k}^x=\mathbf{X}_{i,k+1}^-\boxminus\hat{\mathbf{x}}_{k+1}^- ,
      \qquad
      \mathbf{P}_{k+1}^- =
      \sum_i W_i^{(c)}\mathbf{d}_{i,k}^x(\mathbf{d}_{i,k}^x)^T
      +\mathbf{Q}_{d,k}.

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

   The active sources are selected first. Each state sigma point is then passed
   through the measurement models:

   .. math::

      \mathbf{Z}_{i,k}=h(\mathbf{X}_{i,k}^-,\mathbf{o}_k).

   Additive measurements use a weighted Euclidean mean. Quaternion measurements
   use the same right-error manifold mean as the state:

   .. math::

      \hat{\mathbf{z}}_k \text{ satisfies }
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

   The UKF forms measurement deviations in the residual coordinates and state
   deviations in the tangent coordinates:

   .. math::

      \mathbf{d}_{i,k}^z=\mathbf{Z}_{i,k}\boxminus\hat{\mathbf{z}}_k,
      \qquad
      \mathbf{P}_{zz}=\sum_i W_i^{(c)}\mathbf{d}_{i,k}^z
      (\mathbf{d}_{i,k}^z)^T+\mathbf{R}_k,

   .. math::

      \mathbf{P}_{xz}=\sum_i W_i^{(c)}\mathbf{d}_{i,k}^x
      (\mathbf{d}_{i,k}^z)^T.

   This is the unscented equivalent of constructing a measurement Jacobian;
   the UKF does not calculate an explicit :math:`\mathbf{H}`.

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

   **5. Compute the unscented gain and correction**

   The :class:`~ADCS.covariance.Covariance` operation
   :meth:`~ADCS.covariance.Covariance.updated_unscented` solves the measurement
   covariance and cross-covariance system:

   .. math::

      \mathbf{K}_k=\mathbf{P}_{xz}\mathbf{P}_{zz}^{-1},
      \qquad
      \mathbf{r}_k=\mathbf{z}_k\boxminus\hat{\mathbf{z}}_k,
      \qquad
      \delta\mathbf{x}_k=\mathbf{K}_k\mathbf{r}_k.

   The posterior covariance before moving the tangent origin is the weighted
   state covariance returned by the same operation.

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

   **6. Retract the nominal state and reset the covariance**

   The correction is applied in the selected right-error chart:

   .. math::

      \mathbf{x}_k^+=\mathbf{x}_k^-\boxplus\delta\mathbf{x}_k,
      \qquad
      \mathbf{q}_k^+=\hat{\mathbf{q}}_k^-
      \otimes\phi^{-1}(\delta\boldsymbol{\theta}_k).

   Since the local tangent origin has moved, the posterior covariance is
   transported by the chart-specific reset Jacobian:

   .. math::

      \mathbf{P}_k^+=
      \mathbf{J}_{reset}\mathbf{P}_{k,raw}^+\mathbf{J}_{reset}^T.

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

import numpy as np

from ADCS.covariance import Covariance
from ADCS.estimators.process_model import propagate_state
from ADCS.estimators.process_noise import discretize_process_noise
from ADCS.helpers.math_helpers import quat_diff, quat_mult
from ADCS.state import EstimatorState, State

from .attitude_estimator import AttitudeEstimator


__all__ = ["UKF"]


class UKF(AttitudeEstimator):
    r"""Right-error unscented Kalman attitude estimator.

    Sigma points always span the tangent state covariance.  When an actuator
    has nonzero input-noise covariance, prediction additionally augments the
    sigma points with that actuator-input error and propagates each point with
    ``control + control_error``.  Perfect actuator channels are omitted from
    the augmented dimension.  Continuous state/process noise remains additive
    after propagation, as do measurement-noise covariances.
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
            state,
            dt=dt,
            covariance_coordinates="tangent",
            correction_mode=quaternion_mode,
            measurement_quaternion_mode=quaternion_mode,
            unmodeled_dynamics_psd=unmodeled_dynamics_psd,
        )
        self.alpha = self._finite_positive(alpha, "alpha")
        self.beta = self._finite(beta, "beta")
        self.kappa = self._finite(kappa, "kappa")
        if self._scale(self._state.covariance.dimension) <= 0.0:
            raise ValueError("alpha and kappa must give a positive UKF scale")

    @staticmethod
    def _finite(value: float, name: str) -> float:
        result = float(value)
        if not np.isfinite(result):
            raise ValueError(f"{name} must be finite")
        return result

    @classmethod
    def _finite_positive(cls, value: float, name: str) -> float:
        result = cls._finite(value, name)
        if result <= 0.0:
            raise ValueError(f"{name} must be positive")
        return result

    def _scale(self, dimension: int) -> float:
        return self.alpha**2 * (dimension + self.kappa)

    def _weights(
        self, state: EstimatorState, *, dimension: int | None = None
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Return chart-valid sigma spread and standard unscented weights."""
        dimension = state.covariance.dimension if dimension is None else int(dimension)
        if dimension < state.covariance.dimension:
            raise ValueError("UKF sigma-point dimension cannot be smaller than state dimension")
        scale = self._scale(dimension)
        if scale <= 0.0 or not np.isfinite(scale):
            raise ValueError("alpha and kappa must give a finite positive UKF scale")
        gamma = np.sqrt(scale)
        if self.correction_mode == "quaternion_vector":
            attitude = state.slice("attitude", coordinates="tangent")
            unit_offsets = state.covariance.sigma_offsets()
            largest_attitude_offset = float(
                np.max(np.linalg.norm(unit_offsets[:, attitude], axis=1))
            )
            if largest_attitude_offset:
                # The quaternion-vector retraction is defined only for norms
                # up to two. Keep the sigma points clear of its singular edge.
                gamma = min(gamma, 1.9 / largest_attitude_offset)
                scale = gamma**2
        lam = scale - dimension
        mean = np.full(2 * dimension + 1, 0.5 / scale)
        covariance = mean.copy()
        mean[0] = lam / scale
        covariance[0] = mean[0] + 1.0 - self.alpha**2 + self.beta
        return gamma, mean, covariance

    def _sigma_states(
        self, state: EstimatorState
    ) -> tuple[list[EstimatorState], np.ndarray, np.ndarray, np.ndarray]:
        gamma, mean_weights, covariance_weights = self._weights(state)
        offsets = state.covariance.sigma_offsets(gamma)
        points = [state] + [
            state.plus(
                offset,
                quaternion_mode=self.correction_mode,
                quaternion_order="right",
            )
            for offset in offsets
        ]
        return points, offsets, mean_weights, covariance_weights

    def _control_noise_covariance(self) -> tuple[np.ndarray, np.ndarray]:
        """Return non-perfect command channels and their input covariance.

        The satellite owns actuator uncertainty.  Retaining only channels with
        nonzero marginal variance avoids adding duplicate, zero-offset sigma
        points for perfect actuators while preserving every nonzero covariance
        block (including correlated noisy channels).
        """
        covariance = self.satellite.control_covariance().as_matrix()
        expected = (self.satellite.control_len, self.satellite.control_len)
        if covariance.shape != expected:
            raise ValueError(
                "satellite control covariance must have shape "
                f"{expected}, got {covariance.shape}"
            )
        active = np.flatnonzero(np.diag(covariance) > 0.0)
        return active, covariance[np.ix_(active, active)]

    def _prediction_sigma_points(
        self, state: EstimatorState, control: np.ndarray
    ) -> tuple[
        list[EstimatorState], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """Build state/control sigma points for one prediction.

        State and control errors are independent, so their joint covariance is
        block diagonal.  Control offsets are embedded only in noisy command
        channels; this is equivalent to a full augmented UKF with zero-noise
        channels removed.
        """
        active_controls, control_covariance = self._control_noise_covariance()
        state_dimension = state.covariance.dimension
        augmented_dimension = state_dimension + active_controls.size
        gamma, mean_weights, covariance_weights = self._weights(
            state, dimension=augmented_dimension
        )

        state_offsets = state.covariance.sigma_offsets(gamma)
        points = [state] + [
            state.plus(
                offset,
                quaternion_mode=self.correction_mode,
                quaternion_order="right",
            )
            for offset in state_offsets
        ]
        controls = [control.copy() for _ in points]

        if active_controls.size:
            control_offsets = Covariance(
                control_covariance, coordinates="control"
            ).sigma_offsets(gamma)
            for offset in control_offsets:
                perturbed_control = control.copy()
                perturbed_control[active_controls] += offset
                points.append(state)
                controls.append(perturbed_control)
        else:
            control_offsets = np.zeros((0, 0))

        return (
            points,
            np.asarray(controls),
            state_offsets,
            control_offsets,
            mean_weights,
            covariance_weights,
        )

    def _state_mean(
        self, points: list[EstimatorState], weights: np.ndarray
    ) -> EstimatorState:
        mean = points[0]
        for _ in range(32):
            deviations = np.vstack(
                [
                    point.minus(
                        mean,
                        quaternion_mode=self.correction_mode,
                        quaternion_order="right",
                    )
                    for point in points
                ]
            )
            correction = weights @ deviations
            if np.linalg.norm(correction) <= 1.0e-12:
                return mean
            mean = mean.plus(
                correction,
                quaternion_mode=self.correction_mode,
                quaternion_order="right",
            )
        raise RuntimeError("UKF state mean did not converge")

    def _measurement_mean(
        self,
        stack: Any,
        measurements: list[np.ndarray],
        active: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        mean = measurements[0].copy()
        for entry, enabled in zip(stack.entries, active):
            if not enabled:
                continue
            values = np.vstack(
                [measurement[entry.raw_slice] for measurement in measurements]
            )
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"UKF sigma-point prediction for {entry.name} contains non-finite values"
                )
            if not entry.is_quaternion_attitude:
                mean[entry.raw_slice] = weights @ values
                continue

            quaternion = values[0]
            for _ in range(32):
                deviations = np.vstack(
                    [
                        State.quaternion_delta_to_vector(
                            quat_diff(quaternion, value),
                            mode=self.measurement_quaternion_mode,
                        )
                        for value in values
                    ]
                )
                correction = weights @ deviations
                if np.linalg.norm(correction) <= 1.0e-12:
                    break
                quaternion = quat_mult(
                    quaternion,
                    State.quaternion_delta_from_vector(
                        correction, mode=self.measurement_quaternion_mode
                    ),
                )
                quaternion = quaternion / np.linalg.norm(quaternion)
            else:
                raise RuntimeError(f"UKF measurement mean for {entry.name} did not converge")
            mean[entry.raw_slice] = quaternion
        return mean

    def predict(
        self,
        control: Any,
        orbital_state_start: Any,
        orbital_state_end: Any,
        *,
        dt: float | None = None,
        midpoint_orbital_state: Any | None = None,
    ) -> EstimatorState:
        """Propagate tangent-state sigma points and add discretized process noise."""
        step = self.dt if dt is None else float(dt)
        if not np.isfinite(step) or step < 0.0:
            raise ValueError("dt must be finite and non-negative")
        control = np.array(control, dtype=float, copy=True)
        expected_control_shape = (self.satellite.control_len,)
        if control.shape != expected_control_shape:
            raise ValueError(
                f"control must have shape {expected_control_shape}, got {control.shape}"
            )

        prior = self._state
        (
            points,
            sigma_controls,
            offsets,
            control_offsets,
            mean_weights,
            covariance_weights,
        ) = self._prediction_sigma_points(prior, control)
        propagated_points = [
            propagate_state(
                point,
                self.satellite,
                sigma_control,
                step,
                orbital_state_start,
                orbital_state_end,
                midpoint_orbital_state=midpoint_orbital_state,
            )
            for point, sigma_control in zip(points, sigma_controls)
        ]
        predicted = self._state_mean(propagated_points, mean_weights)
        transition, process_noise = discretize_process_noise(
            prior,
            self.satellite,
            control,
            orbital_state_start,
            step,
            final_state=predicted,
            unmodeled_dynamics_psd=self.unmodeled_dynamics_psd,
            quaternion_mode=self.correction_mode,
            quaternion_order="right",
        )
        deviations = np.vstack(
            [
                point.minus(
                    predicted,
                    quaternion_mode=self.correction_mode,
                    quaternion_order="right",
                )
                for point in propagated_points
            ]
        )
        predicted.covariance = prior.covariance.predicted_unscented(
            deviations, covariance_weights, process_noise
        )
        predicted.process_noise = Covariance(
            process_noise,
            form=prior.process_noise.form,
            coordinates=prior.process_noise.coordinates,
            psd_policy=prior.process_noise.psd_policy,
        )
        self._state = predicted
        self._diagnostics.update(
            transition=transition,
            process_noise=process_noise,
            sigma_offsets=offsets,
            control_noise_covariance=self._control_noise_covariance()[1],
            active_control_indices=np.flatnonzero(
                np.diag(self.satellite.control_covariance().as_matrix()) > 0.0
            ),
            control_noise_offsets=control_offsets,
            prediction_sigma_controls=sigma_controls,
            sigma_weights_mean=mean_weights,
            sigma_weights_covariance=covariance_weights,
            predicted_sigma_deviations=deviations,
            predicted_covariance=predicted.covariance.as_matrix(),
        )
        return self.state

    def correct(
        self,
        measurements: Any,
        orbital_state: Any,
        *,
        enabled: Any | None = None,
        time_s: float | None = None,
        epoch_s: float = 0.0,
    ) -> EstimatorState:
        """Apply a non-augmented unscented measurement update."""
        stack = self.satellite.measurement_stack
        candidate = stack.active_mask(
            measurements, enabled=enabled, time_s=time_s, epoch_s=epoch_s
        )
        nominal_measurement = stack.predict(
            self._state, orbital_state, active_mask=candidate
        )
        active = stack.active_mask(
            measurements,
            enabled=candidate,
            time_s=time_s,
            epoch_s=epoch_s,
            predicted=nominal_measurement,
        )
        prior = self._state
        covariance_size = prior.covariance.dimension
        if not np.any(active):
            self._diagnostics.update(
                active_mask=active,
                predicted_measurement=nominal_measurement,
                innovation=np.empty(0),
                measurement_noise=np.zeros((0, 0)),
                innovation_covariance=np.zeros((0, 0)),
                kalman_gain=np.zeros((covariance_size, 0)),
                correction=np.zeros(covariance_size),
                reset_jacobian=np.eye(covariance_size),
                corrected_covariance=prior.covariance.as_matrix(),
            )
            return self.state

        points, offsets, mean_weights, covariance_weights = self._sigma_states(prior)
        sigma_measurements = [
            stack.predict(point, orbital_state, active_mask=active) for point in points
        ]
        predicted_measurement = self._measurement_mean(
            stack, sigma_measurements, active, mean_weights
        )
        state_deviations = np.vstack(
            [
                point.minus(
                    prior,
                    quaternion_mode=self.correction_mode,
                    quaternion_order="right",
                )
                for point in points
            ]
        )
        measurement_deviations = np.vstack(
            [
                stack.residual(
                    measurement,
                    predicted_measurement,
                    active,
                    quaternion_mode=self.measurement_quaternion_mode,
                )
                for measurement in sigma_measurements
            ]
        )
        measurement_noise = stack.covariance(
            prior,
            active,
            quaternion_mode=self.measurement_quaternion_mode,
        )
        innovation_covariance = Covariance.from_weighted_deviations(
            measurement_deviations,
            covariance_weights,
            measurement_noise,
            form=prior.covariance.form,
            coordinates="measurement_residual",
            psd_policy=prior.covariance.psd_policy,
        )
        gain, posterior_covariance = prior.covariance.updated_unscented(
            state_deviations,
            measurement_deviations,
            covariance_weights,
            measurement_noise,
        )
        innovation = stack.residual(
            measurements,
            predicted_measurement,
            active,
            quaternion_mode=self.measurement_quaternion_mode,
        )
        correction = gain @ innovation
        reset_jacobian = prior.retraction_jacobian(
            correction,
            quaternion_mode=self.correction_mode,
            quaternion_order="right",
        )
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
        self._state = corrected
        self._diagnostics.update(
            active_mask=active,
            predicted_measurement=predicted_measurement,
            innovation=innovation,
            measurement_noise=measurement_noise.as_matrix(),
            innovation_covariance=innovation_covariance.as_matrix(),
            kalman_gain=gain,
            correction=correction,
            reset_jacobian=reset_jacobian,
            sigma_offsets=offsets,
            sigma_weights_mean=mean_weights,
            sigma_weights_covariance=covariance_weights,
            measurement_sigma_deviations=measurement_deviations,
            corrected_covariance=corrected.covariance.as_matrix(),
        )
        return self.state
