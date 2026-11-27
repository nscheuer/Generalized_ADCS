from dataclasses import dataclass, field
from typing import cast

from .estimator.attitude_estimator.base import AttitudeEstimator
from .magnetorquer import Magnetorquer
from .reaction_wheel import ReactionWheel
from .sensors.sensor import Sensor
from .state_machine import ADCSStateMachine


@dataclass(slots=True)
class ADCSSubsystem:
    reaction_wheels: list[ReactionWheel] = field(
        default_factory=lambda: cast(list[ReactionWheel], [])
    )
    magnetorquers: list[Magnetorquer] = field(
        default_factory=lambda: cast(list[Magnetorquer], [])
    )
    sensors: list[Sensor] = field(default_factory=lambda: cast(list[Sensor], []))
    N: int = 1
    estimator: AttitudeEstimator | None = None
    state_machine: ADCSStateMachine | None = None

    def __post_init__(self) -> None:
        if isinstance(self.N, bool) or self.N < 1:
            raise ValueError("N must be a positive integer")
