__all__ = ["AnisotropicNoise"]

import numpy as np
from typing import Optional, Sequence, Union
from numpy.typing import NDArray

from ADCS.satellite_hardware.errors import Noise

class AnisotropicNoise(Noise):
    def __init__(
        self,
        std_cross: float,
        std_roll: float,
        R_noise: Optional[NDArray[np.float64]] = None,
        bounds: Sequence[Union[NDArray, float]] = (-np.array([np.inf]), np.array([np.inf]))
    ) -> None:
        std_aligned = np.array([std_cross, std_cross, std_roll])

        super().__init__(noise=np.zeros(3), std_noise=std_aligned, bounds=bounds)

        self.std_cross = std_cross
        self.std_roll = std_roll

        self._R_noise = R_noise if R_noise is not None else np.eye(3)

    def align_to_body(self, R: NDArray[np.float64]) -> None:
        self._R_noise = R

    def copy(self) -> "AnisotropicNoise":
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
    