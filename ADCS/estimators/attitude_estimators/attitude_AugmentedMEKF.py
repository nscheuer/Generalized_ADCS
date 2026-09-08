"""Multiplicative EKF with joint bias and disturbance-parameter estimation."""

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

