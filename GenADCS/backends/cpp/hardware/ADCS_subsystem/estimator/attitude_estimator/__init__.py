from ._attitude_estimator import AttitudeEstimator, AttitudeEKF, AttitudeMEKF, AttitudeSequentialUKF, Sensor, State
from ._attitude_estimator import predict, correct, correct_gyro, correct_star_tracker, step
from ._attitude_estimator import sample_ekf, sample_ukf, ukf_predict, ukf_step

__all__ = ["AttitudeEstimator", "AttitudeMEKF", "AttitudeEKF", "AttitudeSequentialUKF", "Sensor", "State", "predict", "correct", "correct_gyro", "correct_star_tracker", "step", "sample_ekf", "sample_ukf", "ukf_predict", "ukf_step"]

# Native objects retain the same simulation configuration round-trip as Python.
import copyreg


def _restore_estimator(cls, q, w, p, a, record):
    return cls(q, w, p, a, record=record)


def _reduce_estimator(value):
    return (_restore_estimator, (type(value), *value.prepare(), value.record))


def _restore_ukf(q, w, p, a, alpha, beta, kappa, record):
    return AttitudeSequentialUKF(q, w, p, a, alpha=alpha, beta=beta, kappa=kappa, record=record)


def _reduce_ukf(value):
    return (_restore_ukf, (*value.prepare(), value.alpha, value.beta, value.kappa, value.record))


copyreg.pickle(AttitudeEstimator, _reduce_estimator)
copyreg.pickle(AttitudeMEKF, _reduce_estimator)
copyreg.pickle(AttitudeEKF, _reduce_estimator)
copyreg.pickle(AttitudeSequentialUKF, _reduce_ukf)

from ._attitude_estimator import sample
