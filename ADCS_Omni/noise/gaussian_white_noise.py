from dataclasses import dataclass

from .noise import Noise


@dataclass(slots=True)
class GaussianWhiteNoise(Noise):
    standard_deviation: float
