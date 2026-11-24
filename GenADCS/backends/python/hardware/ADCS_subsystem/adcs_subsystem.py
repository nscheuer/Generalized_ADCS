from dataclasses import dataclass, field

from .magnetorquer import Magnetorquer
from .reaction_wheel import ReactionWheel
from .sensors import Sensor


@dataclass(slots=True)
class ADCSSubsystem:
    reaction_wheels: list[ReactionWheel] = field(default_factory=list)
    magnetorquers: list[Magnetorquer] = field(default_factory=list)
    sensors: list[Sensor] = field(default_factory=list)
