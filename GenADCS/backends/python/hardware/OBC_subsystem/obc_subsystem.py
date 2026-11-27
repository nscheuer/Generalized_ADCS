"""On-Board Computer (OBC) subsystem holding the mission state machine and FDIR."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, cast

from .fdir import FDIR
from .state_machine import StateMachine


@dataclass(slots=True)
class OBCSubsystem:
    """On-Board Computer executing the state machine and FDIR every ``N`` steps."""

    state_machine: StateMachine | None = None
    fdir: FDIR | None = None
    N: int = 1
    timeline: list[tuple[float, str] | tuple[float, str, Any]] = field(
        default_factory=lambda: cast(list[tuple[float, str] | tuple[float, str, Any]], [])
    )

    def __post_init__(self) -> None:
        if isinstance(self.N, bool) or self.N < 1:
            raise ValueError("N must be a positive integer")
        for entry in self.timeline:
            if len(entry) not in (2, 3):
                raise ValueError("timeline entries must be (time, command) or (time, command, goal)")
            t, cmd = entry[0], entry[1]
            if not math.isfinite(t) or t < 0.0:
                raise ValueError("timeline times must be finite and nonnegative")
            if not cmd:
                raise ValueError("timeline telecommand names must be nonempty")
