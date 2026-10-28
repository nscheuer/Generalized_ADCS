r"""Invariant-error EKF with estimated spacecraft hardware parameters.

The attitude error is the SO(3) logarithm of ``R_true R_hat.T`` by default.
Angular velocity, wheel momentum, actuator biases, sensor biases, and
disturbance parameters have additive errors. The augmented transition and
process noise come from the shared spacecraft model, which includes their
coupling to the attitude and angular velocity dynamics.
"""

from __future__ import annotations

from .attitude_IEKF import IEKF

__all__ = ["AugmentedIEKF"]


class AugmentedIEKF(IEKF):
    """IEKF for the full :class:`~ADCS.state.EstimatorState` layout.

    The estimated parameter blocks are Euclidean factors in the filter state.
    Their random-walk noise and process/measurement Jacobians are supplied by
    ``EstimatedSatellite`` and the shared estimator plumbing. The invariant
    attitude error and its measurement-frame transformations are inherited
    from :class:`IEKF`.
    """

    supports_augmented_parameters = True
