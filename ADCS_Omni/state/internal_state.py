from dataclasses import dataclass, field


@dataclass(slots=True)
class InternalState:
    wheel_momentum: object = field(default_factory=list)
    other: dict = field(default_factory=dict)
