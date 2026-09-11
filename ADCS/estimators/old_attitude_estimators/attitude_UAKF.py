"""Compatibility stub for the removed additive unscented attitude filter.

The estimator API was replaced by the eight estimators in
``ADCS.estimators.attitude_estimators``.  ``UAKF`` remains as an import alias
for the non-augmented ``UKF``; callers should select the replacement that
matches their state and covariance configuration.
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

UAKF = UKF

__all__ = [
    "UAKF",
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
