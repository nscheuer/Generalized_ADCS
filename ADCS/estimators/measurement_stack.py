"""Canonical measurement assembly for attitude estimators.

``MeasurementStack`` is the one estimator-facing owner of measurement order,
availability, residual coordinates, covariance, and Jacobians.  Raw telemetry
remains in the satellite's historical order: attitude sensors followed by
reaction-wheel momentum measurements.  The residual vector may have a
different dimension: a quaternion attitude measurement is reduced from four
stored coefficients to three local attitude coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ADCS.covariance import Covariance
from ADCS.helpers.math_helpers import quat_diff
from ADCS.satellite_hardware.sensors import StarTrackerQuaternion
from ADCS.state import EstimatorState, State


__all__ = ["MeasurementStack"]


@dataclass(frozen=True, slots=True)
class _MeasurementEntry:
    """One contiguous source in the canonical raw measurement vector."""

    source: Any
    name: str
    raw_slice: slice
    residual_slice: slice
    sensor_index: int | None = None
    wheel_index: int | None = None

    @property
    def is_quaternion_attitude(self) -> bool:
        return isinstance(self.source, StarTrackerQuaternion)


class MeasurementStack:
    r"""Assemble estimator measurements from an ``EstimatedSatellite``.

    The stack has one entry for each attitude sensor and one entry for each
    reaction-wheel momentum measurement.  An entry is active only when all of
    its raw values are finite, it is explicitly enabled, and, when ``time_s``
    is supplied, its ``sample_time`` schedule is due.  This deliberately makes
    a partially missing vector measurement unavailable as a whole: retaining a
    subset would silently change that sensor's physical measurement model.

    ``active_mask`` is entry-wise, while ``active_measurements`` and
    ``residual`` return compact vectors in the selected entry order.

    Quaternion residuals, Jacobians, and covariances consistently use right
    attitude errors. This is intentionally fixed rather than exposed as a
    partially supported convention knob.
    """

    _SCHEDULE_ATOL_S = 1e-7

    def __init__(self, satellite: Any) -> None:
        self.satellite = satellite
        entries: list[_MeasurementEntry] = []
        raw_offset = 0
        residual_offset = 0

        for index, sensor in enumerate(satellite.attitude_sensors):
            raw_length = int(sensor.output_length)
            residual_length = 3 if isinstance(sensor, StarTrackerQuaternion) else raw_length
            if isinstance(sensor, StarTrackerQuaternion) and sensor.estimate_bias:
                raise ValueError(
                    "StarTrackerQuaternion biases cannot be estimated as additive "
                    "four-coefficient states; use a three-coordinate attitude error model."
                )
            entries.append(
                _MeasurementEntry(
                    source=sensor,
                    name=f"sensor[{index}]",
                    raw_slice=slice(raw_offset, raw_offset + raw_length),
                    residual_slice=slice(residual_offset, residual_offset + residual_length),
                    sensor_index=index,
                )
            )
            raw_offset += raw_length
            residual_offset += residual_length

        for index, wheel in enumerate(satellite.rw_actuators):
            entries.append(
                _MeasurementEntry(
                    source=wheel,
                    name=f"reaction_wheel[{index}]",
                    raw_slice=slice(raw_offset, raw_offset + 1),
                    residual_slice=slice(residual_offset, residual_offset + 1),
                    wheel_index=index,
                )
            )
            raw_offset += 1
            residual_offset += 1

        self._entries = tuple(entries)
        self.raw_size = raw_offset
        self.residual_size = residual_offset

    @property
    def entries(self) -> tuple[_MeasurementEntry, ...]:
        """Ordered measurement sources, primarily for diagnostics and tests."""
        return self._entries

    @property
    def source_order(self) -> tuple[str, ...]:
        """Stable, human-readable order of the measurement sources."""
        return tuple(entry.name for entry in self._entries)

    def readings(self, state: State, orbital_state: Any, *, dmode: Any = None) -> np.ndarray:
        """Return raw readings in the stack's canonical order.

        This is the estimator-neutral telemetry path.  Estimators should pass
        its result to :meth:`active_mask` before performing an update.
        """
        readings = [
            np.atleast_1d(entry.source.reading(state, orbital_state, dmode=dmode))
            if entry.sensor_index is not None
            else np.atleast_1d(entry.source.measure_momentum())
            for entry in self._entries
        ]
        return np.concatenate(readings) if readings else np.empty(0)

    def active_mask(
        self,
        measurements: Any,
        *,
        time_s: float | None = None,
        enabled: Sequence[bool] | None = None,
        epoch_s: float = 0.0,
        predicted: Any | None = None,
    ) -> np.ndarray:
        """Return the active entry mask for a raw measurement vector.

        An entry is also inactive when ``predicted`` is supplied and any of
        its predicted values are non-finite. ``time_s`` is elapsed seconds on
        the sensor sampling timeline; it is
        intentionally not an orbital J2000 value.  When omitted, a received
        finite measurement is treated as live, which is the useful default for
        asynchronous telemetry streams.  A sensor with ``sample_time <= 0`` is
        treated as continuously sampled.  Reaction-wheel measurements are
        continuously sampled until the hardware model gains a sampling period.
        """
        measurements = self._raw_measurements(measurements)
        predicted_values = (
            None if predicted is None else self._raw_measurements(predicted, name="predicted")
        )
        enabled_mask = self._enabled_mask(enabled)
        active = np.zeros(len(self._entries), dtype=bool)
        for index, entry in enumerate(self._entries):
            values = measurements[entry.raw_slice]
            active[index] = enabled_mask[index] and np.all(np.isfinite(values))
            if active[index] and predicted_values is not None:
                active[index] = np.all(np.isfinite(predicted_values[entry.raw_slice]))
            if active[index] and time_s is not None:
                active[index] = self._is_due(entry, time_s, epoch_s)
        return active

    def active_measurements(self, measurements: Any, active_mask: Any) -> np.ndarray:
        """Return the compact raw measurement vector selected by ``active_mask``."""
        measurements = self._raw_measurements(measurements)
        active = self._entry_mask(active_mask)
        parts = [measurements[entry.raw_slice] for entry, selected in zip(self._entries, active) if selected]
        return np.concatenate(parts) if parts else np.empty(0)

    def predict(
        self,
        state: EstimatorState,
        orbital_state: Any,
        active_mask: Any | None = None,
    ) -> np.ndarray:
        """Evaluate raw predicted measurements ``h(x)`` in canonical order.

        When ``active_mask`` is supplied, inactive sensor models are not
        evaluated and their raw slots are filled with NaNs. Pass the result
        back to :meth:`active_mask` as ``predicted`` to remove entries that
        are unavailable at the estimated state.
        """
        self._validate_state(state)
        active = (
            np.ones(len(self._entries), dtype=bool)
            if active_mask is None
            else self._entry_mask(active_mask)
        )
        prediction: list[np.ndarray] = []
        for entry, selected in zip(self._entries, active):
            raw_length = entry.raw_slice.stop - entry.raw_slice.start
            if not selected:
                prediction.append(np.full(raw_length, np.nan))
                continue
            if entry.sensor_index is None:
                prediction.append(np.atleast_1d(state.h[entry.wheel_index]))
                continue
            value = np.atleast_1d(entry.source.clean_reading(state, orbital_state)).astype(float)
            bias = self._sensor_bias(state, entry.sensor_index)
            if bias is not None:
                value = value + bias
            prediction.append(value)
        return np.concatenate(prediction) if prediction else np.empty(0)

    def residual(
        self,
        measurements: Any,
        predicted: Any,
        active_mask: Any,
        *,
        quaternion_mode: str = State.DEFAULT_QUATERNION_MODE,
    ) -> np.ndarray:
        r"""Return compact innovations in their proper local coordinates.

        Additive sources use ``z - h(x)``.  A quaternion star tracker uses the
        right relative quaternion ``q_pred^{-1} * q_measured`` and converts it
        to the three attitude coordinates selected by ``quaternion_mode``.
        """
        measurements = self._raw_measurements(measurements)
        predicted = self._raw_measurements(predicted, name="predicted")
        active = self._entry_mask(active_mask)
        parts: list[np.ndarray] = []
        for entry, selected in zip(self._entries, active):
            if not selected:
                continue
            measured = measurements[entry.raw_slice]
            expected = predicted[entry.raw_slice]
            if not np.all(np.isfinite(measured)) or not np.all(np.isfinite(expected)):
                raise ValueError(
                    f"{entry.name} is active but its measurement or prediction is non-finite; "
                    "recompute active_mask(..., predicted=predicted)"
                )
            if entry.is_quaternion_attitude:
                relative = quat_diff(expected, measured)
                parts.append(
                    State.quaternion_delta_to_vector(relative, mode=quaternion_mode)
                )
            else:
                parts.append(measured - expected)
        return np.concatenate(parts) if parts else np.empty(0)

    def covariance(
        self,
        state: EstimatorState,
        active_mask: Any,
        *,
        form: str = "full",
        quaternion_mode: str = State.DEFAULT_QUATERNION_MODE,
    ) -> Covariance:
        """Return active residual covariance ``R`` using right-error coordinates."""
        self._validate_state(state)
        active = self._entry_mask(active_mask)
        blocks: list[np.ndarray] = []
        for entry, selected in zip(self._entries, active):
            if not selected:
                continue
            if entry.sensor_index is None:
                block = entry.source.momentum_measurement_covariance().as_matrix()
            else:
                block = entry.source.measurement_covariance().as_matrix()
            if entry.is_quaternion_attitude:
                reduction = state.tangent_pinv(
                    quaternion_mode=quaternion_mode,
                    quaternion_order="right",
                )[3:6, 3:7]
                block = reduction @ block @ reduction.T
            blocks.append(block)
        return Covariance.block_diagonal(
            blocks,
            form=form,
            coordinates="measurement_residual",
        )

    def jacobian(
        self,
        state: EstimatorState,
        orbital_state: Any,
        active_mask: Any,
        *,
        quaternion_mode: str = State.DEFAULT_QUATERNION_MODE,
    ) -> np.ndarray:
        r"""Return the active right-error EKF Jacobian ``H`` in tangent coordinates."""
        self._validate_state(state)
        active = self._entry_mask(active_mask)
        rows: list[np.ndarray] = []
        tangent_map = state.tangent_map(
            quaternion_mode=quaternion_mode,
            quaternion_order="right",
        )
        base_size = state.full_slices["wheel_momentum"].stop

        for entry, selected in zip(self._entries, active):
            if not selected:
                continue
            if entry.sensor_index is None:
                row = np.zeros((1, state.tangent_size))
                row[0, state.tangent_slices["wheel_momentum"].start + entry.wheel_index] = 1.0
                rows.append(row)
                continue
            if entry.is_quaternion_attitude:
                row = np.zeros((3, state.tangent_size))
                row[:, state.tangent_slices["attitude"]] = np.eye(3)
                rows.append(row)
                continue

            legacy = np.asarray(entry.source.basestate_jac(state, orbital_state), dtype=float)
            output_size = entry.raw_slice.stop - entry.raw_slice.start
            expected = (base_size, output_size)
            # Older sensor models that do not depend on wheel momentum expose
            # only the original seven physical coordinates [w, q].
            if legacy.shape == (7, output_size) and base_size > 7:
                legacy = np.vstack((legacy, np.zeros((base_size - 7, output_size))))
            if legacy.shape != expected:
                raise ValueError(
                    f"{entry.name}.basestate_jac() must have shape {expected}, got {legacy.shape}"
                )
            full = np.zeros((legacy.shape[1], state.full_size))
            full[:, :base_size] = legacy.T
            bias = self.satellite.sensor_bias_slice(entry.sensor_index)
            if bias is not None:
                bias_jacobian = np.asarray(
                    entry.source.bias_jac(state, orbital_state), dtype=float
                )
                expected_bias_shape = (bias.stop - bias.start, output_size)
                if bias_jacobian.shape != expected_bias_shape:
                    raise ValueError(
                        f"{entry.name}.bias_jac() must have shape "
                        f"{expected_bias_shape}, got {bias_jacobian.shape}"
                    )
                full[:, bias] = bias_jacobian.T
            rows.append(full @ tangent_map)
        return np.vstack(rows) if rows else np.zeros((0, state.tangent_size))

    def _sensor_bias(self, state: EstimatorState, sensor_index: int) -> np.ndarray | None:
        bias_slice = self.satellite.sensor_bias_slice(sensor_index)
        if bias_slice is None:
            return None
        sensor_bias_start = state.full_slices["sensor_bias"].start
        local = slice(
            bias_slice.start - sensor_bias_start,
            bias_slice.stop - sensor_bias_start,
        )
        return state.sens_bias[local]

    def _validate_state(self, state: EstimatorState) -> None:
        if not isinstance(state, EstimatorState):
            raise TypeError(f"state must be an EstimatorState, got {type(state).__name__}")
        if state.h.size != len(self.satellite.rw_actuators):
            raise ValueError(
                "EstimatorState wheel-momentum block must match the number of reaction wheels"
            )
        if state.act_bias.size != self.satellite.act_bias_len:
            raise ValueError("EstimatorState actuator-bias block does not match the satellite layout")
        if state.sens_bias.size != self.satellite.att_sens_bias_len:
            raise ValueError("EstimatorState sensor-bias block does not match the satellite layout")

    def _raw_measurements(self, measurements: Any, *, name: str = "measurements") -> np.ndarray:
        values = np.asarray(measurements, dtype=float)
        if values.shape != (self.raw_size,):
            raise ValueError(f"{name} must have shape ({self.raw_size},), got {values.shape}")
        return values

    def _entry_mask(self, active_mask: Any) -> np.ndarray:
        mask = np.asarray(active_mask, dtype=bool)
        if mask.shape != (len(self._entries),):
            raise ValueError(
                f"active_mask must have shape ({len(self._entries)},), got {mask.shape}"
            )
        return mask

    def _enabled_mask(self, enabled: Sequence[bool] | None) -> np.ndarray:
        if enabled is None:
            return np.ones(len(self._entries), dtype=bool)
        return self._entry_mask(enabled)

    @staticmethod
    def _is_due(entry: _MeasurementEntry, time_s: float, epoch_s: float) -> bool:
        sample_time = getattr(entry.source, "sample_time", None)
        if sample_time is None or sample_time <= 0.0:
            return True
        elapsed_s = float(time_s) - float(epoch_s)
        nearest_sample_s = round(elapsed_s / float(sample_time)) * float(sample_time)
        return bool(
            np.isclose(
                elapsed_s,
                nearest_sample_s,
                rtol=0.0,
                atol=MeasurementStack._SCHEDULE_ATOL_S,
            )
        )
