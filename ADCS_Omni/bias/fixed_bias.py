from dataclasses import dataclass

from .bias import Bias


@dataclass(slots=True)
class FixedBias(Bias):
    value: float
