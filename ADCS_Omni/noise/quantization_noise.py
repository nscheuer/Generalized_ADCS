from dataclasses import dataclass

from .noise import Noise


@dataclass(slots=True)
class QuantizationNoise(Noise):
    step_size: float
