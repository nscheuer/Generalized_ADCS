__all__ = ["Noise"]

import numpy as np
from typing import Sequence

from ADCS.estimators.covariance import Covariance

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
    """
    def __init__(self, noise: np.ndarray | float = np.array([0.0]), std_noise: np.ndarray | float = np.array([0.0]), bounds: Sequence[np.ndarray | float] = (-np.array([np.inf]), np.array([np.inf]))) -> None:
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

    def __bool__(self):
        r"""
        Return ``True`` if the noise model is active.

        Notes
        -----
        The noise model is considered *inactive* (i.e., ``False``) when both
        :math:`n_0 = 0` and :math:`\sigma_n = 0`.
        """
        return not (np.all(self.noise == 0.0) and np.all(self.std_noise == 0.0))
    
    def copy(self):
        return Noise(noise=self.noise, std_noise=self.std_noise, bounds=self.bounds)

    def _update_noise(self) -> None:
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
            scale=self.std_noise
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
        
    def cov(self) -> np.ndarray:
        std2 = self.std_noise * self.std_noise

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
    ) -> Covariance:
        """Return this noise model's owned covariance object."""
        return Covariance(self.cov(), form=form, coordinates=coordinates)
        
    def srcov(self) -> float | np.ndarray:
        if self.std_noise.size == 1:
            # scalar square-root covariance
            return np.array([[self.std_noise.item()]])
        else:
            # diagonal matrix of std deviations
            return np.diagflat(self.std_noise)
