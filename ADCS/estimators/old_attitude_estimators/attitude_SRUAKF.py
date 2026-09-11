"""Compatibility stub for the removed square-root additive UKF.

The estimator API was replaced by the eight estimators in
``ADCS.estimators.attitude_estimators``.  ``SRUAKF`` remains as an import
alias for the non-augmented ``SRUKF``; callers should select the replacement
that matches their state and covariance configuration.
"""

from ADCS.estimators.attitude_estimators import (
    AttitudeEstimator,
    AugmentedEKF,
    AugmentedMEKF,
    AugmentedSRUKF,
    AugmentedUKF,
    EKF,
    MEKF,
    SRUKF,
    UKF,
)

SRUAKF = SRUKF

__all__ = [
    "SRUAKF",
    "AttitudeEstimator",
    "AugmentedEKF",
    "AugmentedMEKF",
    "AugmentedSRUKF",
    "AugmentedUKF",
    "EKF",
    "MEKF",
    "SRUKF",
    "UKF",
]
