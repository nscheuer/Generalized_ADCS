from ._attitude_estimator import AttitudeBatchedUKF, AttitudeEstimator, AttitudeEKF, AttitudeMEKF, AttitudeSequentialSRUKF, AttitudeSequentialUKF, Sensor, State
from ._attitude_estimator import predict, correct, correct_gyro, correct_star_tracker, step
from ._attitude_estimator import batched_ukf_predict, batched_ukf_step, sample_batched_ukf, sample_ekf, sample_srukf, sample_ukf, srukf_predict, srukf_step, ukf_predict, ukf_step

__all__ = ["AttitudeEstimator", "AttitudeMEKF", "AttitudeEKF", "AttitudeSequentialUKF", "AttitudeBatchedUKF", "AttitudeSequentialSRUKF", "Sensor", "State", "predict", "correct", "correct_gyro", "correct_star_tracker", "step", "sample_ekf", "sample_ukf", "sample_batched_ukf", "sample_srukf", "ukf_predict", "ukf_step", "batched_ukf_predict", "batched_ukf_step", "srukf_predict", "srukf_step"]

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


def _restore_srukf(q, w, p, a, alpha, beta, kappa, record):
    return AttitudeSequentialSRUKF(q, w, p, a, alpha=alpha, beta=beta, kappa=kappa, record=record)


def _reduce_srukf(value):
    return (_restore_srukf, (*value.prepare(), value.alpha, value.beta, value.kappa, value.record))


def _restore_batched_ukf(q, w, p, a, alpha, beta, kappa, record):
    return AttitudeBatchedUKF(q, w, p, a, alpha=alpha, beta=beta, kappa=kappa, record=record)


def _reduce_batched_ukf(value):
    return (_restore_batched_ukf, (*value.prepare(), value.alpha, value.beta, value.kappa, value.record))


copyreg.pickle(AttitudeEstimator, _reduce_estimator)
copyreg.pickle(AttitudeMEKF, _reduce_estimator)
copyreg.pickle(AttitudeEKF, _reduce_estimator)
copyreg.pickle(AttitudeSequentialUKF, _reduce_ukf)
copyreg.pickle(AttitudeBatchedUKF, _reduce_batched_ukf)
copyreg.pickle(AttitudeSequentialSRUKF, _reduce_srukf)

from ._attitude_estimator import sample
