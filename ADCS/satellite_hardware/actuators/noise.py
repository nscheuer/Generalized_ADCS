__all__ = ["Noise"]

import numpy as np
from typing import Sequence

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
    def __init__(self, noise: float = 0.0, std_noise: float = 0.0, bounds: Sequence[float] = (-np.inf, np.inf)) -> None:
        self.noise = noise
        self.std_noise = std_noise
        self.bounds = bounds

    def __bool__(self):
        r"""
        Return ``True`` if the noise model is active.

        Notes
        -----
        The noise model is considered *inactive* (i.e., ``False``) when both
        :math:`n_0 = 0` and :math:`\sigma_n = 0`.
        """
        return not (self.noise == 0.0 and self.std_noise == 0.0)

    def _update_noise(self) -> None:
        r"""
        Draw a new Gaussian noise sample and apply bounds.

        The internal noise value is updated as

        .. math::
            n \sim \mathcal{N}(n_0,\ \sigma_n^2),
            \qquad n \leftarrow \mathrm{clip}(n,\ n_{\min},\ n_{\max}).

        This method is called internally each time :meth:`get_noise` is invoked.
        """
        """Update actuator noise with a fresh Gaussian sample."""
        self.noise = np.random.normal(
            loc=self.noise,
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
        self._update_noise()
        return self.noise