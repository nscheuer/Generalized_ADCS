import numpy as np
from typing import Sequence

class Noise:
    def __init__(self, noise: float = 0.0, std_noise: float = 0.0, bounds: Sequence[float] = (-np.inf, np.inf)) -> None:
        self.noise = noise
        self.std_noise = std_noise
        self.bounds = bounds

    def __bool__(self):
        return not (self.noise == 0.0 and self.std_noise == 0.0)

    def _update_noise(self) -> None:
        """Update actuator noise with a fresh Gaussian sample."""
        self.noise = np.random.normal(
            loc=self.noise,
            scale=self.std_noise
        )
        self.noise = np.clip(self.noise, self.bounds[0], self.bounds[1])

    def get_noise(self) -> float:
        self._update_noise()
        return self.noise