from ._attitude_estimator import AttitudeEstimator, AttitudeEKF, AttitudeMEKF, Sensor, State
from ._attitude_estimator import predict, correct, correct_gyro, correct_star_tracker, step
from ._attitude_estimator import sample_ekf

__all__ = ["AttitudeEstimator", "AttitudeMEKF", "AttitudeEKF", "Sensor", "State", "predict", "correct", "correct_gyro", "correct_star_tracker", "step", "sample_ekf"]

# Native objects retain the same simulation configuration round-trip as Python.
import copyreg


def _restore_estimator(cls, q, w, p, a, record):
    return cls(q, w, p, a, record=record)


def _reduce_estimator(value):
    return (_restore_estimator, (type(value), *value.prepare(), value.record))


copyreg.pickle(AttitudeEstimator, _reduce_estimator)
copyreg.pickle(AttitudeMEKF, _reduce_estimator)
copyreg.pickle(AttitudeEKF, _reduce_estimator)

from ._attitude_estimator import sample
