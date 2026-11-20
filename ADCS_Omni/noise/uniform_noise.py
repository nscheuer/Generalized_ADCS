from dataclasses import dataclass

from .noise import Noise


@dataclass(slots=True)
class UniformNoise(Noise):
    lower: float
    upper: float
