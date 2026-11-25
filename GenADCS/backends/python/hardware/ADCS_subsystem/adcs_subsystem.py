from dataclasses import dataclass, field

from .estimator import AttitudeEstimator
from .magnetorquer import Magnetorquer
from .reaction_wheel import ReactionWheel
from .sensors import Sensor


@dataclass(slots=True)
class ADCSSubsystem:
    reaction_wheels: list[ReactionWheel] = field(default_factory=list)
    magnetorquers: list[Magnetorquer] = field(default_factory=list)
    sensors: list[Sensor] = field(default_factory=list)
    N: int = 1
    estimator: AttitudeEstimator | None = None

    def __post_init__(self) -> None:
        if isinstance(self.N, bool) or not isinstance(self.N, int) or self.N < 1:
            raise ValueError("N must be a positive integer")
        if self.estimator is not None and not isinstance(
            self.estimator, AttitudeEstimator
        ):
            raise TypeError("estimator must be an AttitudeEstimator")
