__all__ = ["Bias"]

import numpy as np
from typing import Sequence
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.covariance import Covariance

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
    instability_std : float, optional
        Standard deviation :math:`\sigma_g` of the bias instability, the slow
        wander around the constant offset that an Allan-variance plot shows as
        its floor. Modelled as a first-order Gauss-Markov process with
        correlation time ``instability_time``: it decays towards zero and is
        topped up so that its standard deviation stays at :math:`\sigma_g`.
    instability_time : float, optional
        Correlation time :math:`\tau` of the bias instability in seconds
        (required when ``instability_std`` is set).

    The bias at any time is ``offset + walk + instability``: the constant
    value the device was built with (``bias`` at construction), a random walk
    of rate ``std_bias`` that starts at zero, and the instability. ``bias``
    holds the sum.
    """
    def __init__(self, bias: np.ndarray | float = np.array([0.0]), std_bias: np.ndarray | float = np.array([0.0]), bounds: Sequence[np.ndarray | float] = (-np.array([np.inf]), np.array([np.inf])), instability_std: np.ndarray | float = 0.0, instability_time: float | None = None) -> None:
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
        # Posterior uncertainty reported by an estimator.  Keep it separate
        # from ``std_bias``, which is the configured random-walk rate.
        self.estimated_std_bias = np.zeros_like(bias, dtype=float)
        self.std_bias = std_bias
        self.last_bias_time = float('nan')
        self.bounds = (lo, hi)
        # The three parts of the bias. ``bias`` above is their sum, kept as a
        # plain attribute because the rest of the package reads and writes it.
        self.offset = bias.copy()                     # constant: the value the device was built with
        self.walk = np.zeros_like(bias, dtype=float)  # random walk, rate std_bias, starts at zero
        self.instability = np.zeros_like(bias, dtype=float)  # bias instability, see below
        instability_std = np.asarray(instability_std, dtype=float)
        if instability_std.size == 1:
            instability_std = np.full_like(bias, float(instability_std.reshape(-1)[0]), dtype=float)
        if instability_std.shape != bias.shape:
            raise ValueError("instability_std must be a scalar or match the shape of bias.")
        if np.any(~np.isfinite(instability_std)) or np.any(instability_std < 0.0):
            raise ValueError("instability_std must be finite and non-negative.")
        self.instability_std = instability_std
        if np.any(instability_std > 0.0):
            if instability_time is None or not np.isfinite(instability_time) or instability_time <= 0.0:
                raise ValueError("instability_time (the correlation time in seconds) must be positive when instability_std is set.")
            self.instability_time = float(instability_time)
        else:
            self.instability_time = None if instability_time is None else float(instability_time)

    def __bool__(self):
        r"""
        Return ``True`` if the bias model is active.

        Notes
        -----
        The model is considered *inactive* (i.e. returns ``False``)
        when both :math:`b_0 = 0` and :math:`\sigma_b = 0`.
        """
        return not (np.all(self.bias == 0.0) and np.all(self.std_bias == 0.0) and np.all(self.instability_std == 0.0))
    
    def broadcast_to(self, length: int, *, owner: str = "sensor") -> "Bias":
        """Return this model sized for a ``length``-output sensor.

        A scalar model is repeated per axis, a model that already has
        ``length`` entries is copied, and any other size is a configuration
        error reported here, where the owning sensor is known, instead of
        surfacing later as a covariance-shape error inside the filter.
        Subclasses with their own parameterisation are copied untouched.
        """
        size = int(np.asarray(self.std_bias).size)
        if size == length or type(self) is not Bias:
            return self.copy()
        if size != 1:
            raise ValueError(
                f"{owner} bias model has {size} entries for a {length}-output sensor"
            )

        def repeat(value):
            array = np.asarray(value, dtype=float).reshape(-1)
            return np.full(length, array[0]) if array.size == 1 else array.copy()

        return Bias(
            bias=repeat(self.bias),
            std_bias=repeat(self.std_bias),
            bounds=(repeat(self.bounds[0]), repeat(self.bounds[1])),
            instability_std=repeat(self.instability_std),
            instability_time=self.instability_time,
        )

    def copy(self):
        result = Bias(bias=self.bias, std_bias=self.std_bias, bounds=self.bounds, instability_std=self.instability_std, instability_time=self.instability_time)
        result.estimated_std_bias = self.estimated_std_bias.copy()
        return result

    def process_covariance(
        self,
        dt: float = 1.0,
        *,
        form: str = "full",
    ) -> Covariance:
        r"""Return random-walk covariance :math:`Q_b=\mathrm{diag}(\sigma_b^2\Delta t)`."""
        dt = float(dt)
        if not np.isfinite(dt) or dt < 0.0:
            raise ValueError("dt must be finite and non-negative")
        return Covariance(
            self.process_psd().as_matrix() * dt,
            form=form,
            coordinates="bias",
        )

    def process_psd(self, *, form: str = "full") -> Covariance:
        r"""Return continuous random-walk PSD :math:`\operatorname{diag}(\sigma_b^2)`.

        ``std_bias`` is a random-walk standard-deviation rate, not a
        per-step covariance.  Discretizers such as Van Loan therefore use
        this PSD directly and own the timestep conversion.
        """
        # A bias instability of standard deviation s and correlation time tau
        # has the low-frequency power density of a random walk of rate
        # sqrt(2 s^2 / tau); that is how a filter with a random-walk bias
        # state covers it (the usual practice; the first-order Gauss-Markov
        # process is the standard stand-in for the flicker floor).
        rate_squared = self.std_bias * self.std_bias
        if np.any(self.instability_std > 0.0):
            rate_squared = rate_squared + 2.0 * self.instability_std**2 / self.instability_time
        return Covariance(
            np.diagflat(rate_squared),
            form=form,
            coordinates="bias_rate",
        )

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
        self.walk = np.random.normal(loc=self.walk, scale=self.std_bias*np.sqrt(dt_sec))
        if np.any(self.instability_std > 0.0):
            # First-order Gauss-Markov step, exact for any dt: the process
            # decays towards zero and is topped up so its variance stays at
            # instability_std^2.
            decay = np.exp(-dt_sec / self.instability_time)
            self.instability = decay * self.instability + np.random.normal(
                loc=0.0, scale=self.instability_std * np.sqrt(1.0 - decay * decay)
            )
        self.bias = np.clip(self.offset + self.walk + self.instability, self.bounds[0], self.bounds[1])
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
