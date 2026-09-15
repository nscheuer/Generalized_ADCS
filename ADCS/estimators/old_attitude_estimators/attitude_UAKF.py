"""Compatibility stub for the removed additive unscented attitude filter.

The estimator API was replaced by the eight estimators in
``ADCS.estimators.attitude_estimators``.  ``UAKF`` remains as an import alias
for the non-augmented ``UKF``; callers should select the replacement that
matches their state and covariance configuration.
"""

import warnings

warnings.warn(
    "UAKF is a compatibility alias of the new ADCS.UKF, whose constructor is "
    "UKF(satellite, state, *, dt, ...); the legacy est_sat/J2000/x_hat/P_hat/Q_hat/"
    "cross_term/quat_as_vec interface is gone. Migrate to the new estimators.",
    DeprecationWarning,
    stacklevel=2,
)

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
]
