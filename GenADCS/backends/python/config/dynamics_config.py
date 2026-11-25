from dataclasses import dataclass, field


@dataclass(slots=True)
class DynamicsConfig:
    """Configured force and torque model instances for a simulation."""

    forces: list[object] = field(default_factory=list[object])
    torques: list[object] = field(default_factory=list[object])
