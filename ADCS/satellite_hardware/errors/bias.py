__all__ = ["Bias"]

import numpy as np
from typing import Sequence
from ADCS.orbits.universal_constants import TimeConstants

class Bias:
    r"""
    Represents actuator bias modeled as a bounded random walk.

    Parameters
    ----------
    bias : float, optional
        Initial bias value :math:`b_0` (default 0).
    std_bias : float, optional
        Standard deviation rate :math:`\sigma_b` controlling random-walk diffusion (default 0).
    bounds : (float, float), optional
        Clipping range :math:`[b_{\min}, b_{\max}]` applied after updates.
    """
    def __init__(self, bias: np.ndarray | float = np.array([0.0]), std_bias: np.ndarray | float = np.array([0.0]), bounds: Sequence[np.ndarray | float] = (-np.array([np.inf]), np.array([np.inf])), rng: np.random.Generator | None = None) -> None:
        if isinstance(bias, (float, int)):
            bias = np.array([bias])
        else:
            bias = np.asarray(bias, dtype=float)
        if isinstance(std_bias, (float, int)):
            std_bias = np.array([std_bias])
        else:
            std_bias = np.asarray(std_bias, dtype=float)
        lo, hi = bounds
        lo = np.asarray(lo, dtype=float)
        hi = np.asarray(hi, dtype=float)

        try:
            bias, std_bias, lo, hi = np.broadcast_arrays(bias, std_bias, lo, hi)
        except ValueError:
            raise ValueError(
                "bias, std_bias, and bounds must be broadcastable to the same shape."
            )
        
        if np.any(lo > hi):
            raise ValueError("Each lower bound must be <= the corresponding upper bound.")
        
        self.bias = bias
        self.std_bias = std_bias
        self.last_bias_time = float('nan')
        self.bounds = (lo, hi)
        # Optional per-instance random generator (see Noise.rng). None -> global
        # ``np.random`` (legacy). Set per-satellite via ``Satellite.set_rng``.
        self.rng = rng

    def set_rng(self, rng: np.random.Generator | None) -> None:
        r"""Set the random generator used for the bias random walk."""
        self.rng = rng

    def _generator(self):
        r"""Return the active random source (per-instance Generator or ``np.random``)."""
        gen = getattr(self, "rng", None)
        return gen if gen is not None else np.random

    def __bool__(self):
        r"""
        Return ``True`` if the bias model is active.

        Notes
        -----
        The model is considered *inactive* (i.e. returns ``False``)
        when both :math:`b_0 = 0` and :math:`\sigma_b = 0`.
        """
        return not (np.all(self.bias == 0.0) and np.all(self.std_bias == 0.0))
    
    def copy(self):
        return Bias(bias=self.bias, std_bias=self.std_bias, bounds=self.bounds)

    def _update_bias(self, j2000: float) -> None:
        r"""
        Update the bias value using a Gaussian random walk.

        The bias evolves as

        .. math::
            b_{k+1} \sim \mathcal{N}\!\left(b_k,\ \sigma_b^2\,\Delta t\right),

        where :math:`\Delta t` is the elapsed time in seconds since the last update,
        converted from Julian centuries:

        .. math::
            \Delta t = (t_{J2000}^{(k+1)} - t_{J2000}^{(k)}) \times T_\mathrm{century}.

        The updated value is then clipped to :math:`[b_{\min}, b_{\max}]`.

        Parameters
        ----------
        j2000 : float
            Current time in Julian centuries since J2000.
        """
        """Random Walk"""
        if not np.isfinite(self.last_bias_time):
            self.last_bias_time = j2000
            return
        
        dt_centuries = j2000 - self.last_bias_time
        if dt_centuries <= 0:
            return

        dt_sec = dt_centuries * TimeConstants.cent2sec
        self.bias = self._generator().normal(loc=self.bias, scale=self.std_bias*np.sqrt(dt_sec))
        self.bias = np.clip(self.bias, self.bounds[0], self.bounds[1])
        self.last_bias_time = j2000

    def get_bias(self, j2000: float) -> float:
        r"""
        Return the current bias after applying a random-walk update.

        Parameters
        ----------
        j2000 : float
            Current time in Julian centuries since J2000.

        Returns
        -------
        float
            The updated bias value :math:`b_k`.
        """
        if self.bias.size == 1:
            return self.bias.item()
        else:
            return self.bias