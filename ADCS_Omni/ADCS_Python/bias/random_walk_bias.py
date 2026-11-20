from dataclasses import dataclass

from .bias import Bias


@dataclass(slots=True)
class RandomWalkBias(Bias):
    standard_deviation: float
    initial_value: float = 0.0
