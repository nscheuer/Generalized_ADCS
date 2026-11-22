from dataclasses import dataclass

from .bias import Bias


@dataclass(slots=True)
class ScaleFactorError(Bias):
    fraction: float
