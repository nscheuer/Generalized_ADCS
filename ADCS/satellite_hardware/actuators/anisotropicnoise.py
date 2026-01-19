__all__ = ["AnisotropicNoise"]

import numpy as np
from typing import Optional, Sequence
from numpy.typing import NDArray

from ADCS.satellite_hardware.actuators import Noise

class AnisotropicNoise(Noise):
    def __init__(
        self,
        std_cross: float,
        std_roll,
        R_noise: NDArray[np.float64],
        bounds: Sequence[np.ndarray | float] = (-np.array([np.inf]), np.array([np.inf]))
    ) -> None:
        std_aligned = np.array([std_cross, std_cross, std_roll])

        super().__init__(noise=np.array([0.0, 0.0, 0.0]), std_noise=std_aligned, bounds=bounds)

        self._R_noise = R_noise
        self.std_cross = std_cross
        self.std_roll = std_roll

    def __bool__(self) -> bool:
        return super().__bool__()
    
    def copy(self):
        return AnisotropicNoise(
            std_cross=self.std_cross,
            std_roll=self.std_roll,
            R_noise=self._R_noise.copy(),
            bounds=self.bounds
        )
    
    def _update_noise(self) -> None:
        noise_aligned = np.random.normal(loc=0.0, scale=self.std_noise)
        self.noise = self._R_noise @ noise_aligned
        self.noise = np.clip(self.noise, self.bounds[0], self.bounds[1])

    def cov(self) -> np.ndarray:
        cov_aligned = np.diag(self.std_noise**2)
        return self._R_noise @ cov_aligned @ self._R_noise.T
    
    def srcov(self) -> np.ndarray:
        return np.linalg.cholesky(self.cov())
    
    