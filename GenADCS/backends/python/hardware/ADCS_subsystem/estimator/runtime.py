"""Small numeric dispatch boundary shared by the compiled propagators."""

import numpy as np
from numba import njit

from .attitude_estimator import AttitudeMEKF
from .attitude_estimator.attitude_mekf import step as mekf_step
from .attitude_estimator.attitude_mekf import step_with_diagnostics


def prepare_estimator(estimator):
    if estimator is None:
        return (
            0,
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.zeros(3),
            np.zeros((6, 6)),
            np.zeros((3, 3)),
        )
    if isinstance(estimator, AttitudeMEKF):
        return (1, *estimator.prepare())
    raise TypeError(f"unsupported estimator: {type(estimator).__name__}")


@njit(cache=True)
def update_estimator(
    kind,
    q,
    w,
    covariance,
    acceleration_psd,
    dt,
    readings,
    sensor_config,
    flags=0,
    truth_q=None,
    truth_w=None,
    diagnostics=None,
):
    if kind == 1:
        if flags & 1020:
            q_new, w_new, p_new = step_with_diagnostics(
                q,
                w,
                covariance,
                acceleration_psd,
                dt,
                readings,
                sensor_config,
                flags,
                truth_q,
                truth_w,
                diagnostics,
            )
        else:
            q_new, w_new, p_new = mekf_step(
                q, w, covariance, acceleration_psd, dt, readings, sensor_config
            )
        q[:] = q_new
        w[:] = w_new
        covariance[:, :] = p_new
