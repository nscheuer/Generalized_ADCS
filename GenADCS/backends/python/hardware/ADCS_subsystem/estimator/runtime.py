"""Small numeric dispatch boundary shared by the compiled propagators."""

import numpy as np
from numba import njit

from .attitude_estimator import (
    AttitudeBatchedUKF,
    AttitudeEKF,
    AttitudeEstimator,
    AttitudeMEKF,
    AttitudeSequentialSRUKF,
    AttitudeSequentialUKF,
)
from .attitude_estimator.attitude_batched_ukf import step as batched_ukf_step
from .attitude_estimator.attitude_ekf import step as ekf_step
from .attitude_estimator.attitude_mekf import step as mekf_step
from .attitude_estimator.attitude_mekf import step_with_diagnostics
from .attitude_estimator.attitude_sequential_srukf import step as srukf_step
from .attitude_estimator.attitude_sequential_ukf import step as ukf_step


def prepare_estimator(
    estimator: AttitudeEstimator | None,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    if isinstance(estimator, AttitudeEKF):
        return (2, *estimator.prepare())
    if isinstance(estimator, AttitudeSequentialUKF):
        return (3, *estimator.prepare())
    if isinstance(estimator, AttitudeSequentialSRUKF):
        return (4, *estimator.prepare())
    if isinstance(estimator, AttitudeBatchedUKF):
        return (5, *estimator.prepare())
    raise TypeError(f"unsupported estimator: {type(estimator).__name__}")


def prepare_ukf_params(estimator: AttitudeEstimator | None) -> np.ndarray:
    if isinstance(
        estimator, (AttitudeSequentialUKF, AttitudeSequentialSRUKF, AttitudeBatchedUKF)
    ):
        return estimator.ukf_params.copy()
    return np.array([1.0, 2.0, 0.0], dtype=np.float64)


@njit(cache=True)
def update_estimator(
    kind: int,
    q: np.ndarray,
    w: np.ndarray,
    covariance: np.ndarray,
    acceleration_psd: np.ndarray,
    dt: float,
    readings: np.ndarray,
    sensor_config: tuple[np.ndarray, ...],
    flags: int = 0,
    truth_q: np.ndarray | None = None,
    truth_w: np.ndarray | None = None,
    diagnostics: np.ndarray | None = None,
    references=None,
    ukf_params: np.ndarray | None = None,
) -> None:
    if kind == 1:
        if flags & 1020:
            if truth_q is None or truth_w is None or diagnostics is None:
                raise ValueError("diagnostic updates require truth state and storage")
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
                diagnostics, references,
            )
        else:
            q_new, w_new, p_new = mekf_step(
                q, w, covariance, acceleration_psd, dt, readings, sensor_config, references=references
            )
        q[:] = q_new
        w[:] = w_new
        covariance[:, :] = p_new
    elif kind == 5:
        alpha = ukf_params[0] if ukf_params is not None else 1.0
        beta = ukf_params[1] if ukf_params is not None else 2.0
        kappa = ukf_params[2] if ukf_params is not None else 0.0
        if flags & 1020:
            if truth_q is None or truth_w is None or diagnostics is None:
                raise ValueError("diagnostic updates require truth state and storage")
            q_new, w_new, p_new = batched_ukf_step(
                q, w, covariance, acceleration_psd, dt, readings, sensor_config,
                flags, truth_q, truth_w, diagnostics, references, alpha, beta, kappa,
            )
        else:
            q_new, w_new, p_new = batched_ukf_step(
                q, w, covariance, acceleration_psd, dt, readings, sensor_config,
                references=references, alpha=alpha, beta=beta, kappa=kappa,
            )
        q[:] = q_new
        w[:] = w_new
        covariance[:, :] = p_new
    elif kind == 4:
        alpha = ukf_params[0] if ukf_params is not None else 1.0
        beta = ukf_params[1] if ukf_params is not None else 2.0
        kappa = ukf_params[2] if ukf_params is not None else 0.0
        if flags & 1020:
            if truth_q is None or truth_w is None or diagnostics is None:
                raise ValueError("diagnostic updates require truth state and storage")
            q_new, w_new, p_new = srukf_step(
                q, w, covariance, acceleration_psd, dt, readings, sensor_config,
                flags, truth_q, truth_w, diagnostics, references, alpha, beta, kappa,
            )
        else:
            q_new, w_new, p_new = srukf_step(
                q, w, covariance, acceleration_psd, dt, readings, sensor_config,
                references=references, alpha=alpha, beta=beta, kappa=kappa,
            )
        q[:] = q_new
        w[:] = w_new
        covariance[:, :] = p_new
    elif kind == 5:
        alpha = ukf_params[0] if ukf_params is not None else 1.0
        beta = ukf_params[1] if ukf_params is not None else 2.0
        kappa = ukf_params[2] if ukf_params is not None else 0.0
        if flags & 1020:
            if truth_q is None or truth_w is None or diagnostics is None:
                raise ValueError("diagnostic updates require truth state and storage")
            q_new, w_new, p_new = batched_ukf_step(
                q, w, covariance, acceleration_psd, dt, readings, sensor_config,
                flags, truth_q, truth_w, diagnostics, references, alpha, beta, kappa,
            )
        else:
            q_new, w_new, p_new = batched_ukf_step(
                q, w, covariance, acceleration_psd, dt, readings, sensor_config,
                references=references, alpha=alpha, beta=beta, kappa=kappa,
            )
        q[:] = q_new
        w[:] = w_new
        covariance[:, :] = p_new
    elif kind == 3:
        alpha = ukf_params[0] if ukf_params is not None else 1.0
        beta = ukf_params[1] if ukf_params is not None else 2.0
        kappa = ukf_params[2] if ukf_params is not None else 0.0
        if flags & 1020:
            if truth_q is None or truth_w is None or diagnostics is None:
                raise ValueError("diagnostic updates require truth state and storage")
            q_new, w_new, p_new = ukf_step(
                q, w, covariance, acceleration_psd, dt, readings, sensor_config,
                flags, truth_q, truth_w, diagnostics, references, alpha, beta, kappa,
            )
        else:
            q_new, w_new, p_new = ukf_step(
                q, w, covariance, acceleration_psd, dt, readings, sensor_config,
                references=references, alpha=alpha, beta=beta, kappa=kappa,
            )
        q[:] = q_new
        w[:] = w_new
        covariance[:, :] = p_new
    elif kind == 5:
        alpha = ukf_params[0] if ukf_params is not None else 1.0
        beta = ukf_params[1] if ukf_params is not None else 2.0
        kappa = ukf_params[2] if ukf_params is not None else 0.0
        if flags & 1020:
            if truth_q is None or truth_w is None or diagnostics is None:
                raise ValueError("diagnostic updates require truth state and storage")
            q_new, w_new, p_new = batched_ukf_step(
                q, w, covariance, acceleration_psd, dt, readings, sensor_config,
                flags, truth_q, truth_w, diagnostics, references, alpha, beta, kappa,
            )
        else:
            q_new, w_new, p_new = batched_ukf_step(
                q, w, covariance, acceleration_psd, dt, readings, sensor_config,
                references=references, alpha=alpha, beta=beta, kappa=kappa,
            )
        q[:] = q_new
        w[:] = w_new
        covariance[:, :] = p_new
    elif kind == 4:
        alpha = ukf_params[0] if ukf_params is not None else 1.0
        beta = ukf_params[1] if ukf_params is not None else 2.0
        kappa = ukf_params[2] if ukf_params is not None else 0.0
        if flags & 1020:
            if truth_q is None or truth_w is None or diagnostics is None:
                raise ValueError("diagnostic updates require truth state and storage")
            q_new, w_new, p_new = srukf_step(
                q, w, covariance, acceleration_psd, dt, readings, sensor_config,
                flags, truth_q, truth_w, diagnostics, references, alpha, beta, kappa,
            )
        else:
            q_new, w_new, p_new = srukf_step(
                q, w, covariance, acceleration_psd, dt, readings, sensor_config,
                references=references, alpha=alpha, beta=beta, kappa=kappa,
            )
        q[:] = q_new
        w[:] = w_new
        covariance[:, :] = p_new
    elif kind == 5:
        alpha = ukf_params[0] if ukf_params is not None else 1.0
        beta = ukf_params[1] if ukf_params is not None else 2.0
        kappa = ukf_params[2] if ukf_params is not None else 0.0
        if flags & 1020:
            if truth_q is None or truth_w is None or diagnostics is None:
                raise ValueError("diagnostic updates require truth state and storage")
            q_new, w_new, p_new = batched_ukf_step(
                q, w, covariance, acceleration_psd, dt, readings, sensor_config,
                flags, truth_q, truth_w, diagnostics, references, alpha, beta, kappa,
            )
        else:
            q_new, w_new, p_new = batched_ukf_step(
                q, w, covariance, acceleration_psd, dt, readings, sensor_config,
                references=references, alpha=alpha, beta=beta, kappa=kappa,
            )
        q[:] = q_new
        w[:] = w_new
        covariance[:, :] = p_new
    elif kind == 2:
        if flags & 1020:
            if truth_q is None or truth_w is None or diagnostics is None:
                raise ValueError("diagnostic updates require truth state and storage")
            q_new, w_new, p_new = ekf_step(
                q, w, covariance, acceleration_psd, dt, readings, sensor_config,
                flags, truth_q, truth_w, diagnostics, references,
            )
        else:
            q_new, w_new, p_new = ekf_step(
                q, w, covariance, acceleration_psd, dt, readings, sensor_config, references=references
            )
        q[:] = q_new
        w[:] = w_new
        covariance[:, :] = p_new
    elif kind == 5:
        alpha = ukf_params[0] if ukf_params is not None else 1.0
        beta = ukf_params[1] if ukf_params is not None else 2.0
        kappa = ukf_params[2] if ukf_params is not None else 0.0
        if flags & 1020:
            if truth_q is None or truth_w is None or diagnostics is None:
                raise ValueError("diagnostic updates require truth state and storage")
            q_new, w_new, p_new = batched_ukf_step(
                q, w, covariance, acceleration_psd, dt, readings, sensor_config,
                flags, truth_q, truth_w, diagnostics, references, alpha, beta, kappa,
            )
        else:
            q_new, w_new, p_new = batched_ukf_step(
                q, w, covariance, acceleration_psd, dt, readings, sensor_config,
                references=references, alpha=alpha, beta=beta, kappa=kappa,
            )
        q[:] = q_new
        w[:] = w_new
        covariance[:, :] = p_new
    elif kind == 4:
        alpha = ukf_params[0] if ukf_params is not None else 1.0
        beta = ukf_params[1] if ukf_params is not None else 2.0
        kappa = ukf_params[2] if ukf_params is not None else 0.0
        if flags & 1020:
            if truth_q is None or truth_w is None or diagnostics is None:
                raise ValueError("diagnostic updates require truth state and storage")
            q_new, w_new, p_new = srukf_step(
                q, w, covariance, acceleration_psd, dt, readings, sensor_config,
                flags, truth_q, truth_w, diagnostics, references, alpha, beta, kappa,
            )
        else:
            q_new, w_new, p_new = srukf_step(
                q, w, covariance, acceleration_psd, dt, readings, sensor_config,
                references=references, alpha=alpha, beta=beta, kappa=kappa,
            )
        q[:] = q_new
        w[:] = w_new
        covariance[:, :] = p_new
    elif kind == 5:
        alpha = ukf_params[0] if ukf_params is not None else 1.0
        beta = ukf_params[1] if ukf_params is not None else 2.0
        kappa = ukf_params[2] if ukf_params is not None else 0.0
        if flags & 1020:
            if truth_q is None or truth_w is None or diagnostics is None:
                raise ValueError("diagnostic updates require truth state and storage")
            q_new, w_new, p_new = batched_ukf_step(
                q, w, covariance, acceleration_psd, dt, readings, sensor_config,
                flags, truth_q, truth_w, diagnostics, references, alpha, beta, kappa,
            )
        else:
            q_new, w_new, p_new = batched_ukf_step(
                q, w, covariance, acceleration_psd, dt, readings, sensor_config,
                references=references, alpha=alpha, beta=beta, kappa=kappa,
            )
        q[:] = q_new
        w[:] = w_new
        covariance[:, :] = p_new
    elif kind == 3:
        alpha = ukf_params[0] if ukf_params is not None else 1.0
        beta = ukf_params[1] if ukf_params is not None else 2.0
        kappa = ukf_params[2] if ukf_params is not None else 0.0
        if flags & 1020:
            if truth_q is None or truth_w is None or diagnostics is None:
                raise ValueError("diagnostic updates require truth state and storage")
            q_new, w_new, p_new = ukf_step(
                q, w, covariance, acceleration_psd, dt, readings, sensor_config,
                flags, truth_q, truth_w, diagnostics, references, alpha, beta, kappa,
            )
        else:
            q_new, w_new, p_new = ukf_step(
                q, w, covariance, acceleration_psd, dt, readings, sensor_config,
                references=references, alpha=alpha, beta=beta, kappa=kappa,
            )
        q[:] = q_new
        w[:] = w_new
        covariance[:, :] = p_new
    elif kind == 5:
        alpha = ukf_params[0] if ukf_params is not None else 1.0
        beta = ukf_params[1] if ukf_params is not None else 2.0
        kappa = ukf_params[2] if ukf_params is not None else 0.0
        if flags & 1020:
            if truth_q is None or truth_w is None or diagnostics is None:
                raise ValueError("diagnostic updates require truth state and storage")
            q_new, w_new, p_new = batched_ukf_step(
                q, w, covariance, acceleration_psd, dt, readings, sensor_config,
                flags, truth_q, truth_w, diagnostics, references, alpha, beta, kappa,
            )
        else:
            q_new, w_new, p_new = batched_ukf_step(
                q, w, covariance, acceleration_psd, dt, readings, sensor_config,
                references=references, alpha=alpha, beta=beta, kappa=kappa,
            )
        q[:] = q_new
        w[:] = w_new
        covariance[:, :] = p_new
    elif kind == 4:
        alpha = ukf_params[0] if ukf_params is not None else 1.0
        beta = ukf_params[1] if ukf_params is not None else 2.0
        kappa = ukf_params[2] if ukf_params is not None else 0.0
        if flags & 1020:
            if truth_q is None or truth_w is None or diagnostics is None:
                raise ValueError("diagnostic updates require truth state and storage")
            q_new, w_new, p_new = srukf_step(
                q, w, covariance, acceleration_psd, dt, readings, sensor_config,
                flags, truth_q, truth_w, diagnostics, references, alpha, beta, kappa,
            )
        else:
            q_new, w_new, p_new = srukf_step(
                q, w, covariance, acceleration_psd, dt, readings, sensor_config,
                references=references, alpha=alpha, beta=beta, kappa=kappa,
            )
        q[:] = q_new
        w[:] = w_new
        covariance[:, :] = p_new
    elif kind == 5:
        alpha = ukf_params[0] if ukf_params is not None else 1.0
        beta = ukf_params[1] if ukf_params is not None else 2.0
        kappa = ukf_params[2] if ukf_params is not None else 0.0
        if flags & 1020:
            if truth_q is None or truth_w is None or diagnostics is None:
                raise ValueError("diagnostic updates require truth state and storage")
            q_new, w_new, p_new = batched_ukf_step(
                q, w, covariance, acceleration_psd, dt, readings, sensor_config,
                flags, truth_q, truth_w, diagnostics, references, alpha, beta, kappa,
            )
        else:
            q_new, w_new, p_new = batched_ukf_step(
                q, w, covariance, acceleration_psd, dt, readings, sensor_config,
                references=references, alpha=alpha, beta=beta, kappa=kappa,
            )
        q[:] = q_new
        w[:] = w_new
        covariance[:, :] = p_new
