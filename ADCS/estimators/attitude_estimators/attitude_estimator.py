"""Shared base class for new-generation Kalman attitude estimators.

The EKF and MEKF compose the filter-neutral ``State``, ``Covariance``, process-model, and
``MeasurementStack`` operations. They intentionally do not inherit behavior
from the legacy UKF-era ``Attitude_Estimator`` class.
"""

from __future__ import annotations

from typing import Any, get_args

import numpy as np

from ADCS.covariance import Covariance
from ADCS.estimators.process_model import propagate_state
from ADCS.estimators.process_noise import discretize_process_noise
from ADCS.state import EstimatorState, QuaternionMode


__all__ = ["AttitudeEstimator"]

_QUATERNION_MODES = tuple(get_args(QuaternionMode))


class AttitudeEstimator:
    """Shared prediction/correction lifecycle for the first simple filters.

    This generation deliberately estimates only the physical attitude state;
    estimated hardware biases and disturbance parameters remain future work.
    """

    def __init__(
        self,
        satellite: Any,
        state: EstimatorState,
        *,
        dt: float,
        covariance_coordinates: str,
        correction_mode: str,
        measurement_quaternion_mode: str,
        unmodeled_dynamics_psd: Any = 0.0,
    ) -> None:
        if not isinstance(state, EstimatorState):
            raise TypeError(f"state must be an EstimatorState, got {type(state).__name__}")
        dt = float(dt)
        if not np.isfinite(dt) or dt < 0.0:
            raise ValueError("dt must be finite and non-negative")
        if covariance_coordinates not in ("full", "tangent"):
            raise ValueError("covariance_coordinates must be 'full' or 'tangent'")
        self._validate_charts(
            covariance_coordinates, correction_mode, measurement_quaternion_mode
        )

        self.satellite = satellite
        self.dt = dt
        self._covariance_coordinates = covariance_coordinates
        self._correction_mode = correction_mode
        self._measurement_quaternion_mode = measurement_quaternion_mode
        self._validate_state(state)
        self.unmodeled_dynamics_psd = np.array(
            unmodeled_dynamics_psd, dtype=float, copy=True
        )
        self._state = self._normalize_initial_state(state)
        self._diagnostics: dict[str, np.ndarray] = {}
        self._previous_orbital_state: Any | None = None

    @property
    def covariance_coordinates(self) -> str:
        """Coordinates the covariance is kept in: ``"full"`` (EKF) or ``"tangent"`` (MEKF)."""
        return self._covariance_coordinates

    @property
    def correction_mode(self) -> str:
        """Attitude chart used for the transition matrix, the correction, and the reset."""
        return self._correction_mode

    @property
    def measurement_quaternion_mode(self) -> str:
        """Attitude chart used for quaternion residuals and their Jacobians."""
        return self._measurement_quaternion_mode

    @property
    def state(self) -> EstimatorState:
        """Return an owned snapshot of the current estimate."""
        return self._state.copy()

    @property
    def diagnostics(self) -> dict[str, np.ndarray]:
        """Return owned matrices and vectors from the latest operations."""
        return {name: value.copy() for name, value in self._diagnostics.items()}

    def reset(self, state: EstimatorState) -> EstimatorState:
        """Replace the estimate after validating this filter's layout."""
        self._validate_state(state)
        self._state = self._normalize_initial_state(state)
        self._diagnostics = {}
        self._previous_orbital_state = None
        return self.state

    @staticmethod
    def _validate_charts(
        covariance_coordinates: str,
        correction_mode: str,
        measurement_quaternion_mode: str,
    ) -> None:
        """Reject chart combinations that would silently mix conventions.

        The measurement Jacobian is written in ``measurement_quaternion_mode``
        and applied to a covariance kept in ``correction_mode``. With a tangent
        covariance those are only interchangeable when they name the same chart;
        with a full covariance the correction must be the four-element block.
        """
        for name, mode in (
            ("correction_mode", correction_mode),
            ("measurement_quaternion_mode", measurement_quaternion_mode),
        ):
            if mode not in _QUATERNION_MODES:
                raise ValueError(f"{name} must be one of {_QUATERNION_MODES}, got {mode!r}")
        if measurement_quaternion_mode == "full_quaternion":
            raise ValueError(
                "measurement_quaternion_mode must be a three-parameter attitude chart; "
                "quaternion residuals are always minimal"
            )
        if covariance_coordinates == "full":
            if correction_mode != "full_quaternion":
                raise ValueError(
                    "a full covariance must be corrected with "
                    f"correction_mode='full_quaternion', got {correction_mode!r}"
                )
        elif correction_mode == "full_quaternion":
            raise ValueError(
                "a tangent covariance cannot be corrected with correction_mode='full_quaternion'"
            )
        elif correction_mode != measurement_quaternion_mode:
            raise ValueError(
                f"correction_mode {correction_mode!r} and measurement_quaternion_mode "
                f"{measurement_quaternion_mode!r} must match when the covariance is kept "
                "in tangent coordinates"
            )

    def _validate_state(self, state: EstimatorState) -> None:
        """Validate the state/hardware contract shared by reset and construction."""
        if not isinstance(state, EstimatorState):
            raise TypeError(f"state must be an EstimatorState, got {type(state).__name__}")
        state.validate_layout(
            wheel_momentum=len(self.satellite.rw_actuators),
            actuator_bias=self.satellite.act_bias_len,
            sensor_bias=self.satellite.att_sens_bias_len,
            disturbance_parameter=self.satellite.dist_param_len,
        )
        if state.block_size("estimated_parameters"):
            raise NotImplementedError(
                f"{type(self).__name__} does not yet support estimated biases or "
                "disturbance parameters"
            )
        expected_size = state.size(coordinates=self.covariance_coordinates)
        if state.covariance.dimension != expected_size:
            raise ValueError(
                f"{type(self).__name__} requires a {expected_size}x{expected_size} "
                f"{self.covariance_coordinates} covariance, got {state.covariance.shape}"
            )

    def predict(
        self,
        control: Any,
        orbital_state_start: Any,
        orbital_state_end: Any,
        *,
        dt: float | None = None,
        midpoint_orbital_state: Any | None = None,
    ) -> EstimatorState:
        """Propagate the nominal state and covariance one deterministic step."""
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
        predicted = propagate_state(
            prior,
            self.satellite,
            control,
            step,
            orbital_state_start,
            orbital_state_end,
            midpoint_orbital_state=midpoint_orbital_state,
        )
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
        predicted_covariance = prior.covariance.predicted_linear(
            transition, process_noise
        )
        discrete_noise = Covariance(
            process_noise,
            form=prior.process_noise.form,
            coordinates=prior.process_noise.coordinates,
            psd_policy=prior.process_noise.psd_policy,
        )
        predicted.covariance = predicted_covariance
        predicted.process_noise = discrete_noise
        self._state = predicted
        self._diagnostics.update(
            transition=transition,
            process_noise=process_noise,
            predicted_covariance=predicted_covariance.as_matrix(),
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
        """Apply all selected finite measurements at the current state."""
        stack = self.satellite.measurement_stack
        candidate = stack.active_mask(
            measurements,
            enabled=enabled,
            time_s=time_s,
            epoch_s=epoch_s,
        )
        predicted_measurements = stack.predict(
            self._state, orbital_state, active_mask=candidate
        )
        active = stack.active_mask(
            measurements,
            enabled=candidate,
            time_s=time_s,
            epoch_s=epoch_s,
            predicted=predicted_measurements,
        )
        residual = stack.residual(
            measurements,
            predicted_measurements,
            active,
            quaternion_mode=self.measurement_quaternion_mode,
        )
        measurement_jacobian = stack.jacobian(
            self._state,
            orbital_state,
            active,
            quaternion_mode=self.measurement_quaternion_mode,
            coordinates=self.covariance_coordinates,
        )
        measurement_noise = stack.covariance(
            self._state,
            active,
            quaternion_mode=self.measurement_quaternion_mode,
        )

        prior = self._state
        covariance_size = prior.covariance.dimension
        if residual.size:
            innovation_covariance = (
                measurement_jacobian
                @ prior.covariance.as_matrix()
                @ measurement_jacobian.T
                + measurement_noise.as_matrix()
            )
            gain, joseph_covariance = prior.covariance.updated_linear(
                measurement_jacobian,
                measurement_noise,
                joseph=True,
            )
            correction = gain @ residual
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
            self._state = corrected
        else:
            innovation_covariance = np.zeros((0, 0))
            gain = np.zeros((covariance_size, 0))
            correction = np.zeros(covariance_size)
            reset_jacobian = np.eye(covariance_size)

        self._diagnostics.update(
            active_mask=active,
            predicted_measurement=predicted_measurements,
            innovation=residual,
            measurement_jacobian=measurement_jacobian,
            measurement_noise=measurement_noise.as_matrix(),
            innovation_covariance=innovation_covariance,
            kalman_gain=gain,
            correction=correction,
            reset_jacobian=reset_jacobian,
            corrected_covariance=self._state.covariance.as_matrix(),
        )
        return self.state

    def step(
        self,
        control: Any,
        measurements: Any,
        orbital_state_start: Any,
        orbital_state_end: Any,
        *,
        dt: float | None = None,
        midpoint_orbital_state: Any | None = None,
        enabled: Any | None = None,
        time_s: float | None = None,
        epoch_s: float = 0.0,
    ) -> EstimatorState:
        """Run prediction to ``orbital_state_end`` followed by correction there."""
        self.predict(
            control,
            orbital_state_start,
            orbital_state_end,
            dt=dt,
            midpoint_orbital_state=midpoint_orbital_state,
        )
        return self.correct(
            measurements,
            orbital_state_end,
            enabled=enabled,
            time_s=time_s,
            epoch_s=epoch_s,
        )

    def update(self, u: Any, sensors: Any, os: Any) -> EstimatorState:
        """Adapt ``predict``/``correct`` filters to the simulation update protocol.

        The first sample has no preceding orbital state, so it is corrected in
        place.  Each later sample is propagated from the preceding orbital
        state and then corrected at the current one.
        """
        if self._previous_orbital_state is None:
            self._previous_orbital_state = os
            return self.correct(sensors, os)
        orbital_state_start = self._previous_orbital_state
        self._previous_orbital_state = os
        return self.step(
            u,
            sensors,
            orbital_state_start,
            os,
            midpoint_orbital_state=os,
        )

    def _normalize_initial_state(self, state: EstimatorState) -> EstimatorState:
        normalized = state.normalized()
        if self.covariance_coordinates == "full":
            normalizer = state.normalization_jacobian()
            normalized.covariance = state.covariance.transformed(normalizer)
            normalized.process_noise = state.process_noise.transformed(normalizer)
        return normalized
