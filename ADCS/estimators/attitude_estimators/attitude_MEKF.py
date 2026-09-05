"""Multiplicative extended Kalman attitude estimator."""

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
