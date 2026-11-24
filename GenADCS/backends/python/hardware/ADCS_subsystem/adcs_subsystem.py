from dataclasses import dataclass, field

from .magnetorquer import Magnetorquer
from .reaction_wheel import ReactionWheel
from .sensors import Sensor


@dataclass(slots=True)
class ADCSSubsystem:
    reaction_wheels: list[ReactionWheel] = field(default_factory=list)
    magnetorquers: list[Magnetorquer] = field(default_factory=list)
    sensors: list[Sensor] = field(default_factory=list)
    N: int = 1

    def __post_init__(self):
        if isinstance(self.N, bool) or not isinstance(self.N, int) or self.N < 1:
            raise ValueError("N must be a positive integer")
