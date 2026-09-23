"""Compatibility stub for the removed square-root additive UKF.

The estimator API was replaced by the eight estimators in
``ADCS.estimators.attitude_estimators``.  ``SRUAKF`` remains as an import
alias for the non-augmented ``SRUKF``; callers should select the replacement
that matches their state and covariance configuration.
"""

import warnings

warnings.warn(
    "SRUAKF is a compatibility alias of the new ADCS.SRUKF, whose constructor is "
    "SRUKF(satellite, state, *, dt, ...); the legacy est_sat/J2000/x_hat/P_hat/Q_hat/"
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

SRUAKF = SRUKF

__all__ = [
    "SRUAKF",
]
