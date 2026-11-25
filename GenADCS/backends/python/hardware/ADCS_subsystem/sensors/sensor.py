"""Common three-axis sensor configuration.

Vectors are in metres and tesla where applicable. Orientation and misalignment
are scalar-first quaternions rotating body coordinates into sensor coordinates.
"""

import numpy as np


class Sensor:
    """Numba-compatible sensor configuration; noise is a 3x3 covariance."""

    def __init__(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
        misalignment: np.ndarray,
        saturation_value: float,
        noise: np.ndarray,
        bias: np.ndarray,
    ) -> None:
        if (
            position.shape != (3,)
            or orientation.shape != (4,)
            or misalignment.shape != (4,)
        ):
            raise ValueError("position must have 3 components and quaternions 4")
        if noise.shape != (3, 3) or bias.shape != (3,):
            raise ValueError("noise must be a 3x3 covariance and bias a 3-vector")
        if saturation_value <= 0.0:
            raise ValueError("saturation_value must be positive")
        self.position = position
        self.orientation = orientation / np.sqrt(np.dot(orientation, orientation))
        self.misalignment = misalignment / np.sqrt(np.dot(misalignment, misalignment))
        self.saturation_value = saturation_value
        self.noise = noise
        self.bias = bias
        self.enabled = True

    def availability(self) -> bool:
        return self.enabled
