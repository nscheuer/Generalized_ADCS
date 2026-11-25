"""Bounded random-walk bias model."""

from typing import TYPE_CHECKING

import numpy as np
from numba import njit

if TYPE_CHECKING:
    from ADCS.covariance import Covariance

from GenADCS.backends.python.helpers.random.philox import (
    standard_normals,
    validate_seed,
)

DEFAULT_SEED = np.uint64(0)


@njit(cache=True)
def add_bias(value, bias):
    """Add a prepared bias vector within a compiled measurement kernel."""
    return value + bias


@njit(cache=True)
def advance_random_walk_bias(
    current,
    standard_deviation,
    lower,
    upper,
    dt_seconds,
    seed=DEFAULT_SEED,
    tick=0,
    stream=0x80000000,
):
    """Advance a bounded vector random walk by elapsed seconds."""
    if dt_seconds < 0.0:
        raise ValueError("dt_seconds must be nonnegative")
    candidate = current + standard_deviation * np.sqrt(dt_seconds) * standard_normals(
        seed, tick, stream, current.size
    )
    return np.minimum(np.maximum(candidate, lower), upper)


class Bias:
    """Bias with a configurable initial value, random-walk rate, and bounds."""

    def __init__(self, bias=0.0, std_bias=0.0, bounds=(-np.inf, np.inf)):
        bias = np.atleast_1d(np.asarray(bias, dtype=float))
        std_bias = np.atleast_1d(np.asarray(std_bias, dtype=float))
        lo, hi = (np.asarray(bound, dtype=float) for bound in bounds)
        try:
            bias, std_bias, lo, hi = np.broadcast_arrays(bias, std_bias, lo, hi)
        except ValueError as exc:
            raise ValueError(
                "bias, std_bias, and bounds must be broadcastable to the same shape."
            ) from exc
        if np.any(lo > hi):
            raise ValueError(
                "Each lower bound must be <= the corresponding upper bound."
            )
        self.bias = bias.copy()
        self.estimated_std_bias = np.zeros_like(bias, dtype=float)
        self.std_bias = std_bias.copy()
        self.last_bias_time = float("nan")
        self.bounds = (lo.copy(), hi.copy())
        self._random_tick = 0

    def __bool__(self):
        return not (np.all(self.bias == 0.0) and np.all(self.std_bias == 0.0))

    def broadcast_to(self, length: int, *, owner: str = "sensor") -> "Bias":
        if self.std_bias.size == length or type(self) is not Bias:
            return self.copy()
        if self.std_bias.size != 1:
            raise ValueError(
                f"{owner} bias model has {self.std_bias.size} entries for a {length}-output sensor"
            )

        def repeat(value):
            array = np.asarray(value, dtype=float).reshape(-1)
            return np.full(length, array[0]) if array.size == 1 else array.copy()

        return Bias(
            repeat(self.bias),
            repeat(self.std_bias),
            (repeat(self.bounds[0]), repeat(self.bounds[1])),
        )

    def copy(self):
        result = Bias(self.bias, self.std_bias, self.bounds)
        result.estimated_std_bias = self.estimated_std_bias.copy()
        result._random_tick = self._random_tick
        return result

    def process_covariance(
        self, dt: float = 1.0, *, form: str = "full"
    ) -> "Covariance":
        from ADCS.covariance import Covariance

        dt = float(dt)
        if not np.isfinite(dt) or dt < 0.0:
            raise ValueError("dt must be finite and non-negative")
        return Covariance(
            self.process_psd().as_matrix() * dt, form=form, coordinates="bias"
        )

    def process_psd(self, *, form: str = "full") -> "Covariance":
        from ADCS.covariance import Covariance

        return Covariance(
            np.diagflat(self.std_bias * self.std_bias),
            form=form,
            coordinates="bias_rate",
        )

    def _update_bias(self, j2000: float, rng=None, seed=0, stream=0x80000000) -> None:
        j2000 = float(j2000)
        if not np.isfinite(self.last_bias_time):
            self.last_bias_time = j2000
            return
        dt_centuries = j2000 - self.last_bias_time
        if dt_centuries <= 0.0:
            return
        dt_sec = dt_centuries * 36525.0 * 86400.0
        if rng is None:
            draw = standard_normals(
                validate_seed(seed), self._random_tick, stream, self.bias.size
            ).reshape(self.bias.shape)
            self.bias = self.bias + self.std_bias * np.sqrt(dt_sec) * draw
            self._random_tick += 1
        else:
            self.bias = rng.normal(loc=self.bias, scale=self.std_bias * np.sqrt(dt_sec))
        self.bias = np.clip(self.bias, self.bounds[0], self.bounds[1])
        self.last_bias_time = j2000

    def get_bias(self, j2000: float):
        return self.bias.item() if self.bias.size == 1 else self.bias

    def apply(self, value, dt, rng=None):
        value = np.asarray(value, dtype=np.float64)
        if value.shape != self.bias.shape:
            raise ValueError("value and bias must have matching shapes")
        return add_bias(value, self.bias)


__all__ = ["Bias"]
