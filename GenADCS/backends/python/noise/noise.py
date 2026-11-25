"""Noise configuration and cached numerical kernels for compiled simulation."""

import numpy as np
from numba import njit
from numpy.typing import ArrayLike

from GenADCS.backends.python.helpers.random.philox import next_draw, standard_normals

DEFAULT_SEED = np.uint64(0)


@njit(cache=True)
def gaussian_sample(
    cholesky_factor: np.ndarray, seed: int = DEFAULT_SEED, tick: int = 0
) -> np.ndarray:
    """Draw correlated zero-mean Gaussian noise using a prepared Cholesky factor."""
    return cholesky_factor @ standard_normals(seed, tick, 0, cholesky_factor.shape[0])


@njit(cache=True)
def add_gaussian_noise(
    value: np.ndarray,
    cholesky_factor: np.ndarray,
    seed: int = DEFAULT_SEED,
    tick: int = 0,
) -> np.ndarray:
    return value + gaussian_sample(cholesky_factor, seed, tick)


class Noise:
    """Gaussian covariance noise; ``cholesky_factor`` enters compiled kernels."""

    def __init__(self, covariance: ArrayLike | None = None) -> None:
        if covariance is None:
            self.covariance = None
            self.cholesky_factor = None
            return
        covariance = np.asarray(covariance, dtype=np.float64)
        if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
            raise ValueError("covariance must be a square matrix")
        self.cholesky_factor = np.linalg.cholesky(covariance)
        self.covariance = covariance

    def apply(
        self,
        value: ArrayLike,
        dt: float,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        factor = getattr(self, "cholesky_factor", None)
        if factor is None:
            raise NotImplementedError(
                "configure a covariance or use a concrete noise model"
            )
        value = np.asarray(value, dtype=np.float64)
        if value.shape != (factor.shape[0],):
            raise ValueError("value length must match noise covariance")
        if rng is not None:
            return value + factor @ rng.standard_normal(value.size)
        return value + factor @ next_draw(self, value.size)
