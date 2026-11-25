from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


class _WheelWithMomentum(Protocol):
    h: float


class _SubsystemWithWheels(Protocol):
    reaction_wheels: Sequence[_WheelWithMomentum]


@dataclass(slots=True)
class InternalState:
    wheel_momentum: np.ndarray = field(default_factory=lambda: np.empty(0))

    def __post_init__(self) -> None:
        momentum = np.asarray(self.wheel_momentum, dtype=np.float64)
        if momentum.ndim != 1 or not np.all(np.isfinite(momentum)):
            raise ValueError("wheel_momentum must be a finite 1D array")
        self.wheel_momentum = np.ascontiguousarray(momentum)


class WheelMomentumState(InternalState):
    """Wheel state initialized from an ADCSSubsystem's wheel definitions.

    The source wheels may be ADCS.satellite_hardware.actuators.RW instances.
    Their scalar ``h`` values are copied; propagation does not mutate them.
    """

    def __init__(
        self,
        subsystem: _SubsystemWithWheels | None = None,
        *,
        wheel_momentum: object | None = None,
    ) -> None:
        if subsystem is not None and wheel_momentum is not None:
            raise ValueError("provide subsystem or wheel_momentum, not both")
        if subsystem is not None:
            wheel_momentum = [wheel.h for wheel in subsystem.reaction_wheels]
        super().__init__(np.asarray([] if wheel_momentum is None else wheel_momentum))
