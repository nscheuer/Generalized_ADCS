"""Additive EKF with joint bias and disturbance-parameter estimation."""

from __future__ import annotations

from .attitude_EKF import EKF


__all__ = ["AugmentedEKF"]


class AugmentedEKF(EKF):
    """Additive EKF for the full :class:`~ADCS.state.EstimatorState` layout.

    With empty augmented blocks this is equivalent to the existing ``EKF``.
    When the estimated satellite and state provide bias or disturbance
    parameter blocks, the shared full-quaternion covariance includes them in
    the prediction and measurement update.
    """

    supports_augmented_parameters = True

