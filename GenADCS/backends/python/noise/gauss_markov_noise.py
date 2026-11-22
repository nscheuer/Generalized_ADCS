from dataclasses import dataclass

from .noise import Noise


@dataclass(slots=True)
class GaussMarkovNoise(Noise):
    standard_deviation: float
    correlation_time: float
