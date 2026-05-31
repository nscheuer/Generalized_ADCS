from __future__ import annotations

import numpy as np

from ADCS.helpers.math_helpers import quat_inv, quat_mult, quat_to_vec3
from testing.test_estimators.srukf.helpers import (
    make_baseline_sensors,
    make_estimate_guess,
    make_orbital_sequence,
    make_orbital_state,
    make_mtqs,
    make_satellites,
    make_state,
    make_srukf,
    run_sequence,
    seed,
)


VEC_MODE = 6


def run_srukf_regression_sequence(*, alpha: float | None = None):
    seed(23)
    real_sat, est_sat = make_satellites(
        sensors=make_baseline_sensors(),
        estimated_sensors=make_baseline_sensors(),
        actuators=make_mtqs(),
        estimated_actuators=make_mtqs(),
    )
    srukf = make_srukf(est_sat, x_hat=make_estimate_guess(est_sat), dt=50.0, cross_term=False)
    if alpha is not None:
        srukf.al = alpha
    x_true = make_state(w=np.array([1.5e-3, -1.0e-3, 2.0e-3]), q=np.array([0.99, 0.08, -0.05, 0.09]))
    orbit_sequence = make_orbital_sequence(count=21, dt=50.0, base=make_orbital_state())
    return run_sequence(real_sat, srukf, x_true=x_true, os_sequence=orbit_sequence)


def attitude_error_angles(q_true: np.ndarray, q_est: np.ndarray) -> np.ndarray:
    errors = np.zeros((len(q_true), 3))
    for index in range(len(q_true)):
        dq = quat_mult(
            q_true[index] / np.linalg.norm(q_true[index]),
            quat_inv(q_est[index] / np.linalg.norm(q_est[index])),
        )
        if dq[0] < 0:
            dq = -dq
        errors[index] = 2.0 * dq[1:4]
    return errors


def consistency_percentages(result) -> tuple[list[float], list[float]]:
    covariance = np.asarray(result.covariances)
    truth = result.truth
    estimate = result.estimate
    start = len(covariance) // 2 if len(covariance) // 2 != len(covariance) else 0

    rate_sigma = 3.0 * np.sqrt(covariance[:, 0:3, 0:3].diagonal(axis1=1, axis2=2))
    rate_error = np.abs(truth[:, 0:3] - estimate[:, 0:3])
    rate_percentages = [float(np.mean(rate_error[start:, axis] <= rate_sigma[start:, axis])) for axis in range(3)]

    attitude_sigma = 3.0 * np.sqrt(covariance[:, 3:6, 3:6].diagonal(axis1=1, axis2=2))
    attitude_error = np.abs(attitude_error_angles(truth[:, 3:7], estimate[:, 3:7]))
    attitude_percentages = [float(np.mean(attitude_error[start:, axis] <= attitude_sigma[start:, axis])) for axis in range(3)]
    return rate_percentages, attitude_percentages


def nees_metrics(result, *, vec_mode: int = VEC_MODE) -> tuple[np.ndarray, float]:
    covariance = np.asarray(result.covariances)
    truth = result.truth
    estimate = result.estimate
    start = len(covariance) // 2 if len(covariance) // 2 != len(covariance) else 0

    attitude_nees = []
    rate_inside = []
    for index in range(start, len(covariance)):
        dq = quat_mult(
            quat_inv(estimate[index, 3:7] / np.linalg.norm(estimate[index, 3:7])),
            truth[index, 3:7] / np.linalg.norm(truth[index, 3:7]),
        )
        if dq[0] < 0:
            dq = -dq
        error = quat_to_vec3(dq, vec_mode)
        block = covariance[index, 3:6, 3:6]
        try:
            attitude_nees.append(float(error @ np.linalg.solve(block, error)))
        except np.linalg.LinAlgError:
            pass

        rate_sigma = 3.0 * np.sqrt(np.diag(covariance[index, 0:3, 0:3]))
        rate_error = np.abs(truth[index, 0:3] - estimate[index, 0:3])
        rate_inside.append(bool(np.all(rate_error <= rate_sigma)))

    return np.asarray(attitude_nees), float(np.mean(rate_inside))
