__all__ = ["Noise"]

import numpy as np
from typing import Sequence

from ADCS.covariance import Covariance

class Noise:
    r"""
    Represents additive actuator noise with optional Gaussian randomness and bounds.

    Parameters
    ----------
    noise : float, optional
        Mean or nominal noise offset :math:`n_0` (default 0).
    std_noise : float, optional
        Standard deviation :math:`\sigma_n` of the Gaussian perturbation (default 0).
    bounds : (float, float), optional
        Lower and upper limits :math:`[n_{\min}, n_{\max}]` applied after sampling.
    density : float, optional
        White-noise density :math:`\eta`, in reading units per root hertz
        (equivalently reading units times :math:`\sqrt{\mathrm{s}}`). A reading
        averaged over a sample interval :math:`\Delta t` then has standard
        deviation :math:`\eta/\sqrt{\Delta t}`, so the noise scales correctly
        with the sample rate; ``std_noise`` is the part that does not scale
        (quantisation, electronics). For a gyro this is the angle random walk,
        for a magnetometer the noise density of the data sheet. The total
        standard deviation of a reading is
        :math:`\sqrt{\sigma_n^2 + \eta^2/\Delta t}`, see :meth:`std`.
    """
    def __init__(self, noise: np.ndarray | float = np.array([0.0]), std_noise: np.ndarray | float = np.array([0.0]), bounds: Sequence[np.ndarray | float] = (-np.array([np.inf]), np.array([np.inf])), density: np.ndarray | float | None = None) -> None:
        if isinstance(noise, (float, int)):
            noise = np.array([noise])
        else:
            noise = np.asarray(noise, dtype=float)
        if isinstance(std_noise, (float, int)):
            std_noise = np.array([std_noise])
        else:
            std_noise = np.asarray(std_noise, dtype=float)
        lo, hi = bounds
        lo = np.asarray(lo, dtype=float)
        hi = np.asarray(hi, dtype=float)

        try:
            noise, std_noise, lo, hi = np.broadcast_arrays(noise, std_noise, lo, hi)
        except ValueError:
            raise ValueError(
                "noise, std_noise, and bounds must be broadcastable to the same shape."
            )
        
        if np.any(lo > hi):
            raise ValueError("Each lower bound must be <= the corresponding upper bound.")
        
        
        self.noise = noise
        self.std_noise = std_noise
        self.bounds = (lo, hi)
        if density is None:
            density = np.zeros_like(std_noise)
        density = np.asarray(density, dtype=float)
        if density.ndim == 0 or density.size == 1:
            density = np.full_like(std_noise, float(density.reshape(-1)[0]))
        if density.shape != std_noise.shape:
            raise ValueError("density must be a scalar or match the shape of std_noise.")
        if np.any(density < 0.0):
            raise ValueError("density must be non-negative.")
        self.density = density

    def __bool__(self):
        r"""
        Return ``True`` if the noise model is active.

        Notes
        -----
        The noise model is considered *inactive* (i.e., ``False``) when both
        :math:`n_0 = 0` and :math:`\sigma_n = 0`.
        """
        return not (np.all(self.noise == 0.0) and np.all(self.std_noise == 0.0) and np.all(self.density == 0.0))
    
    def broadcast_to(self, length: int, *, owner: str = "sensor") -> "Noise":
        """Return this model sized for a ``length``-output sensor.

        A scalar model is repeated per axis, a model that already has
        ``length`` entries is copied, and any other size is a configuration
        error reported here, where the owning sensor is known, instead of
        surfacing later as a covariance-shape error inside the filter.
        Subclasses with their own parameterisation are copied untouched.
        """
        size = int(np.asarray(self.std_noise).size)
        if size == length or type(self) is not Noise:
            return self.copy()
        if size != 1:
            raise ValueError(
                f"{owner} noise model has {size} entries for a {length}-output sensor"
            )

        def repeat(value):
            array = np.asarray(value, dtype=float).reshape(-1)
            return np.full(length, array[0]) if array.size == 1 else array.copy()

        return Noise(
            noise=repeat(self.noise),
            std_noise=repeat(self.std_noise),
            bounds=(repeat(self.bounds[0]), repeat(self.bounds[1])),
            density=repeat(self.density),
        )

    def copy(self):
        return Noise(noise=self.noise, std_noise=self.std_noise, bounds=self.bounds, density=self.density)

    def std(self, sample_time: float | None = None) -> np.ndarray:
        r"""Standard deviation of one reading taken over ``sample_time`` seconds.

        :math:`\sqrt{\sigma_n^2 + \eta^2/\Delta t}`. The sample time is only
        needed when a density is set: a model without one is the same at every
        rate, and a model with one cannot be evaluated without the rate.
        """
        if np.all(self.density == 0.0):
            return np.asarray(self.std_noise, dtype=float).copy()
        if sample_time is None:
            raise ValueError(
                "this noise model has a density, so its standard deviation depends on the "
                "sample time; pass sample_time (sensors pass their own)"
            )
        sample_time = float(sample_time)
        if not np.isfinite(sample_time) or sample_time <= 0.0:
            raise ValueError("sample_time must be positive and finite when a noise density is set")
        return np.sqrt(self.std_noise**2 + self.density**2 / sample_time)

    def _update_noise(self, sample_time: float | None = None) -> None:
        r"""
        Draw a new Gaussian noise sample and apply bounds.

        The internal noise value is updated as

        .. math::
            n \sim \mathcal{N}(n_0,\ \sigma_n^2),
            \qquad n \leftarrow \mathrm{clip}(n,\ n_{\min},\ n_{\max}).

        This method is called internally each time :math:`get_noise` is invoked.
        """
        """Update actuator noise with a fresh Gaussian sample."""
        self.noise = np.random.normal(
            loc=0.0,
            scale=self.std(sample_time)
        )
        self.noise = np.clip(self.noise, self.bounds[0], self.bounds[1])

    def get_noise(self) -> float:
        r"""
        Return a fresh bounded Gaussian noise sample.

        Returns
        -------
        float
            The updated noise value :math:`n` after random sampling and clipping.
        """
        if self.noise.size == 1:
            return self.noise.item()
        else:
            return self.noise
        
    def cov(self, sample_time: float | None = None) -> np.ndarray:
        std = self.std(sample_time)
        std2 = std * std

        if std2.size == 1:
            # scalar covariance
            return np.array([[std2.item()]])
        else:
            # diagonal covariance matrix
            return np.diagflat(std2)

    def covariance(
        self,
        *,
        form: str = "full",
        coordinates: str = "generic",
        sample_time: float | None = None,
    ) -> Covariance:
        """Return this noise model's owned covariance object."""
        return Covariance(self.cov(sample_time), form=form, coordinates=coordinates)
        
    def srcov(self, sample_time: float | None = None) -> float | np.ndarray:
        std = self.std(sample_time)
        if std.size == 1:
            # scalar square-root covariance
            return np.array([[std.item()]])
        else:
            # diagonal matrix of std deviations
            return np.diagflat(std)
