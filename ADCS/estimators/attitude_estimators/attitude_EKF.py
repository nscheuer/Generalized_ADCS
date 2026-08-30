"""Additive extended Kalman attitude estimator."""

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
