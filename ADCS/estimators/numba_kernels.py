"""Compiled array kernels shared by the modern attitude estimators.

The attitude filters deliberately keep their satellite, state, and measurement
objects in Python: those objects provide validation and allow different hardware
models.  The covariance recursions, however, operate exclusively on homogeneous
``float64`` arrays and are executed for every EKF/UKF update.  Keeping the
kernels here lets all eight modern filters use Numba without coupling their
public APIs to Numba types.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def linear_covariance_prediction(
    transition: np.ndarray, covariance: np.ndarray, noise: np.ndarray
) -> np.ndarray:
    """Return ``F @ P @ F.T + Q``, explicitly symmetrized for roundoff."""
    result = transition @ covariance @ transition.T + noise
    return (result + result.T) * 0.5


@njit(cache=True)
def linear_kalman_update(
    covariance: np.ndarray,
    measurement_jacobian: np.ndarray,
    measurement_noise: np.ndarray,
    joseph: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the gain and posterior covariance of a linear Kalman update."""
    innovation = (
        measurement_jacobian @ covariance @ measurement_jacobian.T
        + measurement_noise
    )
    gain = np.linalg.solve(innovation, measurement_jacobian @ covariance).T
    if joseph:
        residual = np.eye(covariance.shape[0]) - gain @ measurement_jacobian
        posterior = (
            residual @ covariance @ residual.T
            + gain @ measurement_noise @ gain.T
        )
    else:
        posterior = (
            np.eye(covariance.shape[0]) - gain @ measurement_jacobian
        ) @ covariance
    return gain, (posterior + posterior.T) * 0.5


@njit(cache=True)
def weighted_outer_covariance(
    deviations: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Return ``sum_i weights[i] * deviations[i] * deviations[i].T``."""
    dimension = deviations.shape[1]
    result = np.zeros((dimension, dimension))
    for sample in range(deviations.shape[0]):
        weight = weights[sample]
        for row in range(dimension):
            value = weight * deviations[sample, row]
            for column in range(row, dimension):
                result[row, column] += value * deviations[sample, column]
    for row in range(dimension):
        for column in range(row):
            result[row, column] = result[column, row]
    return result


@njit(cache=True)
def weighted_cross_covariance(
    first_deviations: np.ndarray,
    second_deviations: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Return ``sum_i weights[i] * first[i] * second[i].T``."""
    result = np.zeros((first_deviations.shape[1], second_deviations.shape[1]))
    for sample in range(first_deviations.shape[0]):
        weight = weights[sample]
        for row in range(first_deviations.shape[1]):
            value = weight * first_deviations[sample, row]
            for column in range(second_deviations.shape[1]):
                result[row, column] += value * second_deviations[sample, column]
    return result


@njit(cache=True)
def weighted_row_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return the weighted mean of row vectors without temporary broadcasts."""
    result = np.zeros(values.shape[1])
    for sample in range(values.shape[0]):
        for column in range(values.shape[1]):
            result[column] += weights[sample] * values[sample, column]
    return result
