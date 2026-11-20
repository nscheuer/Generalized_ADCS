from dataclasses import dataclass

from .noise import Noise


@dataclass(slots=True)
class FlickerNoise(Noise):
    amplitude: float
