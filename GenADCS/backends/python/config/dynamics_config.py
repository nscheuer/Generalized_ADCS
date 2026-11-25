from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class DynamicsConfig:
    """Configured force and torque model instances for a simulation."""

    forces: list[Any] = field(default_factory=list)
    torques: list[Any] = field(default_factory=list)
