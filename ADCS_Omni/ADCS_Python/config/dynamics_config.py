from dataclasses import dataclass, field


@dataclass(slots=True)
class DynamicsConfig:
    """Configured force and torque model instances for a simulation."""

    forces: list = field(default_factory=list)
    torques: list = field(default_factory=list)
