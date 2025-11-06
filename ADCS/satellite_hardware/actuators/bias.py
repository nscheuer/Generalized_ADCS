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
    def __init__(self, bias: float = 0.0, std_bias: float = 0.0, bounds: Sequence[float] = (-np.inf, np.inf)) -> None:
        self.bias = bias
        self.std_bias = std_bias
        self.last_bias_time = float('nan')
        self.bounds = bounds

    def __bool__(self):
        r"""
        Return ``True`` if the bias model is active.

        Notes
        -----
        The model is considered *inactive* (i.e. returns ``False``)
        when both :math:`b_0 = 0` and :math:`\sigma_b = 0`.
        """
        return not (self.bias == 0.0 and self.std_bias == 0.0)

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

        self.bias = np.random.normal(
            loc=self.bias,
            scale=self.std_bias * np.sqrt(dt_sec)
        )
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
        self._update_bias(j2000=j2000)
        return self.bias