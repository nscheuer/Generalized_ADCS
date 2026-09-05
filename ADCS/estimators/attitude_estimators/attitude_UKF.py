"""Non-augmented unscented Kalman attitude estimator."""

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
    r"""Right-error, non-augmented unscented Kalman attitude estimator.

    Sigma points span only the tangent state covariance.  Continuous process
    noise is discretized and added after state propagation, while measurement
    noise is added after transforming the sigma-point measurements.  In
    particular, this filter does not augment the state with process, control,
    sensor, bias, or disturbance variables.
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
        self, state: EstimatorState
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Return chart-valid sigma spread and standard unscented weights."""
        dimension = state.covariance.dimension
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
        points, offsets, mean_weights, covariance_weights = self._sigma_states(prior)
        propagated_points = [
            propagate_state(
                point,
                self.satellite,
                control,
                step,
                orbital_state_start,
                orbital_state_end,
                midpoint_orbital_state=midpoint_orbital_state,
            )
            for point in points
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
