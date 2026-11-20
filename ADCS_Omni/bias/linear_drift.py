from dataclasses import dataclass

from .bias import Bias


@dataclass(slots=True)
class LinearDrift(Bias):
    rate: float
    initial_value: float = 0.0
