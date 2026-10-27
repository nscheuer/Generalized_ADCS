__all__ = ["BiasWithInstability"]

import numpy as np
from typing import Sequence

from ADCS.covariance import Covariance
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.errors.bias import Bias


class BiasWithInstability(Bias):
    r"""
    A bias that is a constant offset, a random walk **and a bias instability**.

    :class:`~ADCS.satellite_hardware.errors.bias.Bias` models the bias as a
    starting value plus a random walk of rate ``std_bias``. Real gyros and
    magnetometers also show **bias instability**: a slow wander around the
    current value that an Allan-variance plot shows as its floor (the
    "bias instability" line of a data sheet, in deg/h or nT). It is neither
    constant nor a random walk: it stays bounded, with a characteristic
    time over which it decorrelates.

    This subclass adds that term as a first-order Gauss-Markov process with
    standard deviation :math:`\sigma_g` (``instability_std``) and correlation
    time :math:`\tau` (``instability_time``), the standard stand-in for the
    flicker floor:

    .. math::

        g_{k+1} = e^{-\Delta t/\tau} g_k + w_k,
        \qquad
        w_k \sim \mathcal{N}\!\left(0,\ \sigma_g^2 (1 - e^{-2\Delta t/\tau})\right),

    exact for any step :math:`\Delta t`, so the process stays stationary with
    standard deviation :math:`\sigma_g`. The reported bias is the base
    class's random walk plus :math:`g`. Everything the base class does
    (offset, random walk, bounds, estimator write-back through ``bias``)
    is unchanged; only the step and the process PSD differ.

    For a filter with a random-walk bias state, the instability is covered by
    adding its low-frequency power density :math:`2\sigma_g^2\tau` to the
    walk rate squared in :meth:`process_psd`.

    Parameters
    ----------
    bias, std_bias, bounds
        As for :class:`~ADCS.satellite_hardware.errors.bias.Bias`.
    instability_std : float or array
        Standard deviation :math:`\sigma_g` of the bias instability.
    instability_time : float
        Correlation time :math:`\tau` in seconds.
    """

    def __init__(
        self,
        bias: np.ndarray | float = np.array([0.0]),
        std_bias: np.ndarray | float = np.array([0.0]),
        bounds: Sequence[np.ndarray | float] = (-np.array([np.inf]), np.array([np.inf])),
        *,
        instability_std: np.ndarray | float,
        instability_time: float,
    ) -> None:
        super().__init__(bias=bias, std_bias=std_bias, bounds=bounds)
        instability_std = np.atleast_1d(np.asarray(instability_std, dtype=float))
        if instability_std.size == 1:
            instability_std = np.full_like(self.bias, float(instability_std[0]), dtype=float)
        if instability_std.shape != self.bias.shape:
            raise ValueError("instability_std must be a scalar or match the shape of bias.")
        if np.any(~np.isfinite(instability_std)) or np.any(instability_std < 0.0):
            raise ValueError("instability_std must be finite and non-negative.")
        instability_time = float(instability_time)
        if not np.isfinite(instability_time) or instability_time <= 0.0:
            raise ValueError("instability_time (the correlation time in seconds) must be positive.")
        self.instability_std = instability_std
        self.instability_time = instability_time
        self.instability = np.zeros_like(self.bias, dtype=float)  # current value of g

    def __bool__(self):
        return super().__bool__() or bool(np.any(self.instability_std != 0.0))

    def broadcast_to(self, length: int, *, owner: str = "sensor") -> "BiasWithInstability":
        """Size this model for a ``length``-output sensor (see :meth:`Bias.broadcast_to`)."""
        size = int(np.asarray(self.std_bias).size)
        if size == length:
            return self.copy()
        if size != 1:
            raise ValueError(f"{owner} bias model has {size} entries for a {length}-output sensor")

        def repeat(value):
            array = np.asarray(value, dtype=float).reshape(-1)
            return np.full(length, array[0]) if array.size == 1 else array.copy()

        return BiasWithInstability(
            bias=repeat(self.bias),
            std_bias=repeat(self.std_bias),
            bounds=(repeat(self.bounds[0]), repeat(self.bounds[1])),
            instability_std=repeat(self.instability_std),
            instability_time=self.instability_time,
        )

    def copy(self) -> "BiasWithInstability":
        result = BiasWithInstability(
            bias=self.bias, std_bias=self.std_bias, bounds=self.bounds,
            instability_std=self.instability_std, instability_time=self.instability_time,
        )
        result.estimated_std_bias = self.estimated_std_bias.copy()
        result.instability = self.instability.copy()
        result.last_bias_time = self.last_bias_time
        return result

    def process_psd(self, *, form: str = "full") -> Covariance:
        r"""Random-walk PSD plus the instability's low-frequency density :math:`2\sigma_g^2\tau`."""
        rate_squared = self.std_bias * self.std_bias + 2.0 * self.instability_std**2 * self.instability_time
        return Covariance(np.diagflat(rate_squared), form=form, coordinates="bias_rate")

    def _update_bias(self, j2000: float) -> None:
        r"""One random-walk step (as the base class) plus one Gauss-Markov step of the instability."""
        if not np.isfinite(self.last_bias_time):
            self.last_bias_time = j2000
            return
        dt_centuries = j2000 - self.last_bias_time
        if dt_centuries <= 0:
            return
        dt_sec = dt_centuries * TimeConstants.cent2sec
        bias = np.random.normal(loc=self.bias, scale=self.std_bias * np.sqrt(dt_sec))
        if np.any(self.instability_std > 0.0):
            decay = np.exp(-dt_sec / self.instability_time)
            new_instability = decay * self.instability + np.random.normal(
                loc=0.0, scale=self.instability_std * np.sqrt(1.0 - decay * decay)
            )
            bias = bias + (new_instability - self.instability)
            self.instability = new_instability
        self.bias = np.clip(bias, self.bounds[0], self.bounds[1])
        self.last_bias_time = j2000
